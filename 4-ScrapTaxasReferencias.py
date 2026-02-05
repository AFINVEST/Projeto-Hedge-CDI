# -*- coding: utf-8 -*-
# Curva DI x Pré (CSV B3) + Interpolação por DU (log-DF, PCHIP)
# - DOWNLOAD do arquivo da B3 para SPECIFIC_DATE (GetDownloadFile)
#   * Trata resposta que pode vir como: CSV direto, BASE64 do CSV em text/plain,
#     ou JSON/text com URL (fallback)
# - Lê CSV local (robusto: força ; e fallback split)
# - Usa DU do arquivo (SEM recalcular DU)
# - Gera curva diária por DU (datas via busday_offset)
# - Atualiza PARQUET_RAW e gera PARQUET_INTERP

from __future__ import annotations

import re
import json
import base64
import binascii
import unicodedata
from pathlib import Path
from functools import lru_cache
from io import StringIO

import numpy as np
import pandas as pd
import requests

# ===================== CONFIG =====================
DATA_DIR = Path("Dados")
DATA_DIR.mkdir(exist_ok=True)

USE_SPECIFIC_DATE = True
SPECIFIC_DATE = pd.Timestamp("2026-01-30")  # <- ajuste aqui

# fallback local (se quiser rodar sem baixar)
INPUT_CSV = Path(r"Dados\TaxaReferencia_PRE_20251215.csv")

FERIADOS_PATH = Path("feriados_nacionais.xls")

PARQUET_RAW = DATA_DIR / "b3_taxas_ref_di_raw.parquet"
PARQUET_INTERP = DATA_DIR / "curva_di_interpolada_por_DU.parquet"

# B3
RATE_ID = "PRE"
LANGUAGE = "pt-br"
BASE_HOST = "https://sistemaswebb3-derivativos.b3.com.br"
BASE_DOWNLOAD = f"{BASE_HOST}/referenceRatesProxy/Search/GetDownloadFile"

REQ_TIMEOUT = (10, 60)
HEADERS_HTTP = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": f"{BASE_HOST}/",
    "Origin": BASE_HOST,
}

# ===================== HELPERS (download) =====================
def _make_download_token(language: str, date_iso: str, rate_id: str) -> str:
    payload = {"language": language, "date": date_iso, "id": rate_id}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _decode_bytes_best_effort(b: bytes) -> str:
    # ordem importa: B3 frequentemente vem cp1252/latin1
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1", "ISO-8859-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def _seems_like_csv_text(txt: str) -> bool:
    if not txt:
        return False
    t = txt[:8000].replace("\r\n", "\n").replace("\r", "\n")
    low = _strip_accents(t.lower())
    # cabeçalho típico: "Descricao da Taxa;Dias Uteis;Dias Corridos;Preco/Taxa"
    return ("dias uteis" in low) and ("preco/taxa" in low or ("preco" in low and "taxa" in low))


def _looks_like_csv_bytes(b: bytes) -> bool:
    return _seems_like_csv_text(_decode_bytes_best_effort(b))


def _extract_url_from_text(s: str) -> str | None:
    m = re.search(r"https?://[^\s\"'>]+", s or "")
    return m.group(0) if m else None


def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_b64_to_bytes(text: str) -> bytes | None:
    s = (text or "").strip()
    s2 = "".join(s.split())

    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s2 or ""):
        return None

    pad = (-len(s2)) % 4
    if pad:
        s2 = s2 + ("=" * pad)

    try:
        return base64.b64decode(s2, validate=False)
    except (binascii.Error, ValueError):
        return None


def _normalize_possible_url(s: str) -> str | None:
    s = (s or "").strip().strip('"').strip("'")
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("/"):
        return f"{BASE_HOST}{s}"
    return None


def _try_resolve_download_pointer_text(text: str) -> list[str]:
    text = (text or "").strip()
    urls: list[str] = []

    u = _extract_url_from_text(text)
    if u:
        urls.append(u)

    js = _try_parse_json(text)
    if isinstance(js, dict):
        candidates = []
        for k in ("url", "downloadUrl", "downloadURL", "fileUrl", "fileURL", "link", "href", "path"):
            v = js.get(k)
            if isinstance(v, str):
                candidates.append(v)

        for v in js.values():
            if isinstance(v, dict):
                for k in ("url", "downloadUrl", "fileUrl", "link", "href", "path"):
                    vv = v.get(k)
                    if isinstance(vv, str):
                        candidates.append(vv)

        for v in candidates:
            uu = _normalize_possible_url(v)
            if uu:
                urls.append(uu)

    uu = _normalize_possible_url(text)
    if uu:
        urls.append(uu)

    if " " not in text and "\n" not in text and len(text) >= 6 and not text.startswith("{"):
        token = text.strip().strip('"').strip("'")
        urls.extend([
            f"{BASE_HOST}/referenceRatesProxy/Search/Download/{token}",
            f"{BASE_HOST}/referenceRatesProxy/Search/GetDownload/{token}",
            f"{BASE_HOST}/referenceRatesProxy/Download/{token}",
            f"{BASE_HOST}/referenceRatesProxy/Search/DownloadFile/{token}",
        ])

    seen = set()
    out = []
    for x in urls:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def download_taxa_referencia_csv_specific_date(
    ref_date: pd.Timestamp,
    rate_id: str = RATE_ID,
    language: str = LANGUAGE,
    out_dir: Path = DATA_DIR,
    cookies: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    date_iso = ref_date.strftime("%Y-%m-%d")
    ymd = ref_date.strftime("%Y%m%d")

    token = _make_download_token(language, date_iso, rate_id)
    url = f"{BASE_DOWNLOAD}/{token}"

    s = requests.Session()
    s.headers.update(HEADERS_HTTP)
    if cookies:
        s.cookies.update(cookies)

    r = s.get(url, timeout=REQ_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(
            f"Falha no GetDownloadFile ({r.status_code}). "
            f"Content-Type={r.headers.get('content-type')} | trecho={r.text[:300]!r}"
        )

    raw = r.content
    out_path = out_dir / f"TaxaReferencia_{rate_id}_{ymd}.csv"

    # (A) CSV direto
    if _looks_like_csv_bytes(raw):
        out_path.write_bytes(raw)
        print(f"[download ok] CSV direto: {out_path} ({len(raw)} bytes)")
        return out_path

    text = r.text if hasattr(r, "text") else raw.decode("utf-8", errors="ignore")

    # (B) base64 do CSV
    b64_bytes = _try_b64_to_bytes(text)
    if b64_bytes and _looks_like_csv_bytes(b64_bytes):
        out_path.write_bytes(b64_bytes)
        print(f"[download ok] CSV veio em base64 (text/plain): {out_path} ({len(b64_bytes)} bytes)")
        return out_path

    # (C) fallback ponteiro/url
    dbg = out_dir / f"DEBUG_GetDownloadFile_{rate_id}_{ymd}.txt"
    dbg.write_text(text, encoding="utf-8", errors="ignore")

    candidates = _try_resolve_download_pointer_text(text)
    last_err = None

    for u in candidates:
        try:
            rr = s.get(u, timeout=REQ_TIMEOUT)
            if rr.status_code != 200:
                last_err = f"{u} -> {rr.status_code}"
                continue

            content = rr.content
            if _looks_like_csv_bytes(content):
                out_path.write_bytes(content)
                print(f"[download ok] via URL: {u} -> {out_path} ({len(content)} bytes)")
                return out_path

            t2 = rr.text
            bb2 = _try_b64_to_bytes(t2)
            if bb2 and _looks_like_csv_bytes(bb2):
                out_path.write_bytes(bb2)
                print(f"[download ok] via URL->base64: {u} -> {out_path} ({len(bb2)} bytes)")
                return out_path

            last_err = f"{u} -> 200 mas não era CSV (ct={rr.headers.get('content-type')})"

        except Exception as e:
            last_err = f"{u} -> {e}"

    raise RuntimeError(
        "Não consegui chegar no CSV final.\n"
        f"Último erro: {last_err}\n"
        f"Debug do GetDownloadFile salvo em: {dbg}"
    )


# ===================== PARSERS / CSV =====================
def parse_num_br_smart(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s:
        return np.nan

    s = s.replace("%", "").replace("\u00A0", "").replace(" ", "")
    # 14,90 -> 14.90
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


@lru_cache(maxsize=1)
def load_feriados_set() -> set:
    if not FERIADOS_PATH.exists():
        return set()
    try:
        df = pd.read_excel(FERIADOS_PATH)
    except Exception:
        return set()
    if df.empty or len(df.columns) == 0:
        return set()
    df = df.dropna()
    df = df[[df.columns[0]]]
    feriados = (
        pd.to_datetime(df[df.columns[0]], errors="coerce", dayfirst=True)
        .dropna()
        .dt.normalize()
        .dt.date
        .unique()
    )
    return set(feriados.tolist())


def _holidays_np(feriados_set: set) -> np.ndarray:
    return np.array(list(feriados_set), dtype="datetime64[D]") if feriados_set else np.array([], dtype="datetime64[D]")


def _infer_ref_date_from_filename(p: Path) -> pd.Timestamp:
    m = re.search(r"(\d{8})", p.name)
    if not m:
        return pd.Timestamp.today().normalize()
    ymd = m.group(1)
    try:
        return pd.Timestamp(year=int(ymd[0:4]), month=int(ymd[4:6]), day=int(ymd[6:8])).normalize()
    except Exception:
        return pd.Timestamp.today().normalize()


def _read_csv_b3_super_robust(path: Path) -> pd.DataFrame:
    """
    Leitura robusta pra B3:
    1) decodifica bytes (cp1252/latin1 etc)
    2) força sep=';'
    3) se ainda vier 1 coluna, faz split manual por ';'
    """
    raw = path.read_bytes()
    txt = _decode_bytes_best_effort(raw)

    # normaliza quebras
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = [ln for ln in txt.split("\n") if ln.strip()]

    if not lines:
        return pd.DataFrame()

    # tenta achar a linha de header (a primeira que contenha "dias" e algum delimitador)
    header_idx = 0
    for i, ln in enumerate(lines[:20]):
        low = _strip_accents(ln.lower())
        if "dias" in low and (";" in ln or "\t" in ln or "," in ln):
            header_idx = i
            break

    txt2 = "\n".join(lines[header_idx:])

    # 1) força ';'
    df = pd.read_csv(StringIO(txt2), sep=";", engine="python")
    # se veio 1 coluna, tenta split
    if df.shape[1] == 1:
        col0 = df.columns[0]
        # tenta split do header
        if ";" in col0:
            header = [c.strip() for c in col0.split(";")]
            body = df[col0].astype(str).str.split(";", expand=True)
            body.columns = header[: body.shape[1]]
            df = body
        else:
            # último fallback: tentar tab
            df = pd.read_csv(StringIO(txt2), sep="\t", engine="python")

    # limpa colunas
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def load_b3_csv_taxa_ref(path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    df = _read_csv_b3_super_robust(path)

    # DEBUG se não parece correto
    ymd = _infer_ref_date_from_filename(path).strftime("%Y%m%d")
    dbg = DATA_DIR / f"DEBUG_parse_csv_{RATE_ID}_{ymd}.txt"

    if df.empty:
        dbg.write_text(f"DF vazio\n\nTop 40 linhas:\n{path.read_text(errors='ignore')[:4000]}\n", encoding="utf-8", errors="ignore")
        return pd.DataFrame(columns=["DU","dias_corridos","di_aa_252","pre_aa_360"]), _infer_ref_date_from_filename(path)

    cols_norm = {c: _strip_accents(c.lower()) for c in df.columns}

    def find_col(predicate):
        for c, cn in cols_norm.items():
            if predicate(cn):
                return c
        return None

    col_du = find_col(lambda x: ("dias uteis" in x) or (x == "du"))
    col_dc = find_col(lambda x: ("dias corridos" in x) or ("corridos" in x))

    # >>> FIX: não pode pegar "descricao da taxa" como taxa numérica <<<
    col_tax = find_col(
        lambda x: (
            ("preco/taxa" in x)                              # caso padrão B3
            or (("preco" in x) and ("taxa" in x))            # variações
            or (("taxa" in x) and ("preco" in x))            # redundante (ok)
            or (("taxa" in x) and ("valor" in x))            # se algum dia mudar
        )
        and ("descricao" not in x)                           # ignora "descricao da taxa"
    )

    if col_du is None or col_tax is None:
        dbg.write_text(
            "Não consegui identificar colunas.\n"
            f"Columns: {list(df.columns)}\n\n"
            f"Head:\n{df.head(20).to_string(index=False)}\n",
            encoding="utf-8",
            errors="ignore",
        )
        raise ValueError(f"CSV lido mas colunas inesperadas. Colunas: {list(df.columns)}")

    out = pd.DataFrame()
    out["DU"] = pd.to_numeric(df[col_du], errors="coerce").astype("Int64")

    if col_dc is not None:
        out["dias_corridos"] = pd.to_numeric(df[col_dc], errors="coerce").astype("Int64")
    else:
        out["dias_corridos"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    out["di_aa_252"] = df[col_tax].map(parse_num_br_smart)
    out["pre_aa_360"] = np.nan

    out = out.dropna(subset=["DU", "di_aa_252"]).copy()
    out["DU"] = out["DU"].astype(int)
    out = out.sort_values("DU").drop_duplicates("DU").reset_index(drop=True)

    ref_dt = _infer_ref_date_from_filename(path)

    # se ficou vazio, salva debug bem detalhado
    if out.empty:
        dbg.write_text(
            "Após parse, OUT ficou vazio.\n"
            f"Columns: {list(df.columns)}\n"
            f"col_du={col_du} | col_tax={col_tax} | col_dc={col_dc}\n\n"
            f"Head original:\n{df.head(30).to_string(index=False)}\n",
            encoding="utf-8",
            errors="ignore",
        )

    return out, ref_dt


# ===================== INTERPOLAÇÃO (log-DF sobre DU) =====================
try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except Exception:
    PchipInterpolator = None
    _HAVE_SCIPY = False


def construir_interpolador_di_por_DU_from_file(df_taxas: pd.DataFrame):
    du_nodes = df_taxas["DU"].to_numpy(dtype=float)
    r_aa = (df_taxas["di_aa_252"].to_numpy(dtype=float) / 100.0)

    if len(du_nodes) == 0:
        raise ValueError("Sem nós para interpolação (df_taxas vazio).")

    if du_nodes[0] > 0:
        du_nodes = np.insert(du_nodes, 0, 0.0)
        r_aa = np.insert(r_aa, 0, r_aa[0])

    u = du_nodes / 252.0
    DF_nodes = (1.0 + r_aa) ** (-u)
    y = np.log(DF_nodes)

    if _HAVE_SCIPY:
        f_logdf = PchipInterpolator(u, y, extrapolate=True)
    else:
        def f_logdf(x):
            x = np.asarray(x, float)
            return np.interp(x, u, y, left=y[0], right=y[-1])

    def _to_u(du_val):
        return np.asarray(du_val, float) / 252.0

    def di_aa_at_du(du_val):
        uu = _to_u(du_val)
        DF = np.exp(f_logdf(uu))
        out = np.empty_like(uu)

        near0 = np.isclose(uu, 0.0)
        not0 = ~near0

        out[near0] = r_aa[0]
        out[not0] = np.power(DF[not0], -1.0 / uu[not0]) - 1.0
        return out * 100.0

    def di_daily_at_du(du_val):
        rr = di_aa_at_du(du_val) / 100.0
        return np.power(1.0 + rr, 1.0 / 252.0) - 1.0

    return {
        "di_aa_at_du": di_aa_at_du,
        "di_daily_at_du": di_daily_at_du,
        "du_max": int(np.nanmax(du_nodes)),
    }


def gerar_curva_diaria_por_DU(interp, ref_dt: pd.Timestamp, feriados_np: np.ndarray) -> pd.DataFrame:
    du_grid = np.arange(0, interp["du_max"] + 1, dtype=int)
    ref_np = np.datetime64(ref_dt.normalize().date(), "D")
    dates_np = np.busday_offset(ref_np, du_grid, roll="forward", holidays=feriados_np)
    dates_pd = pd.to_datetime(dates_np.astype("datetime64[D]"))

    di_aa = interp["di_aa_at_du"](du_grid)
    di_d = interp["di_daily_at_du"](du_grid)

    return pd.DataFrame(
        {
            "ref_date": ref_dt.normalize(),
            "data": dates_pd,
            "DU": du_grid,
            "di_aa_252_interp_pct": di_aa,
            "di_diaria_interp": di_d,
        }
    )


# ===================== PIPELINE: CSV -> RAW + INTERP =====================
def run_from_csv_and_interpolate(csv_path: Path):
    df_day, ref_dt = load_b3_csv_taxa_ref(csv_path)

    if df_day.empty:
        raise RuntimeError(
            "Parse do CSV gerou df_day vazio. "
            "Veja o arquivo DEBUG_parse_csv_*.txt na pasta Dados."
        )

    add = df_day.copy()
    add["ref_date"] = ref_dt
    add["source"] = "csv_b3_download"
    add = add[["ref_date", "DU", "dias_corridos", "di_aa_252", "pre_aa_360", "source"]]

    if PARQUET_RAW.exists():
        try:
            base = pd.read_parquet(PARQUET_RAW)
        except OSError as exc:
            backup = PARQUET_RAW.with_suffix(
                PARQUET_RAW.suffix + f".corrupt.{pd.Timestamp.now():%Y%m%d_%H%M%S}"
            )
            PARQUET_RAW.rename(backup)
            print(f"[warn] parquet corrompido, movido para: {backup} | erro: {exc}")
            base = add
        else:
            base = base[base["ref_date"] != ref_dt]

            # evita warning de concat com vazio
            if not add.empty:
                base = pd.concat([base, add], ignore_index=True)
    else:
        base = add

    base.sort_values(["ref_date", "DU"]).to_parquet(PARQUET_RAW, index=False)

    fer_np = _holidays_np(load_feriados_set())
    interp = construir_interpolador_di_por_DU_from_file(df_day)
    curva_du = gerar_curva_diaria_por_DU(interp, ref_dt, fer_np)
    curva_du.to_parquet(PARQUET_INTERP, index=False)

    print(f"[ok] ref_date={ref_dt.date()} | nós={len(df_day)} | RAW={PARQUET_RAW} | INTERP={PARQUET_INTERP}")
    return ref_dt, df_day, curva_du


# ===================== MAIN =====================
if __name__ == "__main__":
    if USE_SPECIFIC_DATE:
        cookies = None
        csv_path = download_taxa_referencia_csv_specific_date(
            SPECIFIC_DATE, rate_id=RATE_ID, language=LANGUAGE, out_dir=DATA_DIR, cookies=cookies
        )
        ref_dt, df_today, curva_du = run_from_csv_and_interpolate(csv_path)
    else:
        ref_dt, df_today, curva_du = run_from_csv_and_interpolate(INPUT_CSV)

    print("\n--- Resumo ---")
    print("ref_date:", ref_dt.date())
    print(df_today.head())
    print(curva_du.head())
