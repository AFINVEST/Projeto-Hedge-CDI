from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
import pandas as pd
import time
import re

ativos = ["HAPV21", "KLBNA2"]

service = Service()
driver = webdriver.Chrome(service=service)
driver.maximize_window()
driver.execute_script("document.body.style.zoom='60%'")

df = pd.DataFrame()

def parse_vne_int(vna_text: str) -> int:
    """
    Ex: 'R$ 500,000000' -> 500 (int)
    """
    if not vna_text:
        return None
    s = vna_text.strip()

    # mantém apenas números, separadores e sinal
    s = re.sub(r"[^\d\.,\-]", "", s)

    # remove separador de milhar (.) e troca decimal (,) por (.)
    s = s.replace(".", "").replace(",", ".")

    try:
        return int(round(float(s)))
    except:
        return None

def parse_taxa_emissao_decimal(rem_text: str) -> float:
    """
    Ex: '110,5500% DI' -> 1.1055 (float)
    """
    if not rem_text:
        return None

    s = rem_text.strip()

    # pega o trecho antes do %
    if "%" in s:
        s = s.split("%")[0].strip()

    # remove tudo que não seja número, ponto, vírgula ou sinal
    s = re.sub(r"[^\d\.,\-]", "", s)

    # remove separador de milhar (.) e troca decimal (,) por (.)
    s = s.replace(".", "").replace(",", ".")

    try:
        pct = float(s)          # 110.5500
        return pct / 100.0      # 1.1055
    except:
        return None

def extrair_vne_e_taxa_emissao(wait: WebDriverWait):
    """
    Após clicar em 'Calcular' e a página renderizar os resultados,
    extrai:
      - VNA (salvar como VNE, inteiro)
      - Remuneração (TaxaEmissao, decimal)
    """
    # VNA: <dt>...VNA...</dt> e <dd>R$ ...</dd>
    vna_dd = wait.until(EC.presence_of_element_located((
        By.XPATH,
        "//dl[.//dt[contains(., 'VNA')]]//dd"
    )))
    vna_text = vna_dd.text.strip()
    vne = parse_vne_int(vna_text)

    # Remuneração: <dl id="calc-remuneracao"> ... <dd>110,5500% DI</dd>
    rem_dd = wait.until(EC.presence_of_element_located((
        By.CSS_SELECTOR,
        "dl#calc-remuneracao dd"
    )))
    rem_text = rem_dd.text.strip()
    taxa_emissao = parse_taxa_emissao_decimal(rem_text)

    return vne, taxa_emissao


try:
    for ativo in ativos:
        driver.get(f"https://data.anbima.com.br/ferramentas/calculadora/debentures/{ativo}?ativo=debentures")
        wait = WebDriverWait(driver, 20)
        driver.execute_script("document.body.style.zoom='60%'")

        # checa se aparece o texto "Taxa ANBIMA do ativo" (se sim, dá pra calcular direto)
        time.sleep(2)
        try:
            driver.find_element(By.XPATH, "//p[contains(text(), 'Taxa ANBIMA do ativo')]")
            taxa_anbima_encontrada = True
        except:
            taxa_anbima_encontrada = False

        if taxa_anbima_encontrada:
            # clicar em calcular
            button = wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "#card-calcular-precificacao > article > article > section > div > form > "
                "div.col-xs-12.precificacao-content__calculate-button.col-no-padding > button"
            )))
            button.click()

            # aguarda tabela do fluxo
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "#card-fluxo-pagamento > article > article > section > div > div > table"
            )))

            # extrai VNE (VNA) e TaxaEmissao (Remuneração)
            vne, taxa_emissao = extrair_vne_e_taxa_emissao(wait)

            # parse da tabela com bs4
            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.select_one("#card-fluxo-pagamento > article > article > section > div > div > table")
            rows = table.find_all("tr")

            data_list = []
            for row in rows:
                columns = row.find_all("td")
                data = [col.text.strip() for col in columns]
                if data:
                    data_list.append(data)

            df_append = pd.DataFrame(data_list)
            df_append["Ativo"] = ativo
            df_append["VNE"] = vne
            df_append["TaxaEmissao"] = taxa_emissao

            df = pd.concat([df, df_append], ignore_index=True)

        else:
            # busca taxa na página de características
            driver.get(f"https://data.anbima.com.br/debentures/{ativo}/caracteristicas")
            driver.execute_script("document.body.style.zoom='60%'")
            wait = WebDriverWait(driver, 20)

            try:
                taxa_elemento = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "p.lower-card-item-value")))
                taxa_texto = taxa_elemento.text  # ex: "14,830000 %"
                taxa_valor = float(taxa_texto.replace(" %", "").replace(",", "."))

                # volta pra calculadora
                driver.get(f"https://data.anbima.com.br/ferramentas/calculadora/debentures/{ativo}?ativo=debentures")
                wait = WebDriverWait(driver, 20)
                driver.execute_script("document.body.style.zoom='60%'")

                # preenche o input da taxa
                taxa_formatada = str(f"{taxa_valor:.6f}").replace(".", ",")

                input_elemento = wait.until(EC.presence_of_element_located((
                    By.XPATH, "//div[@id='precificacao-input-taxa']//input"
                )))

                driver.execute_script("""
                    const input = arguments[0];
                    const valor = arguments[1];
                    input.value = valor;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                """, input_elemento, taxa_formatada)

                input_elemento.send_keys(Keys.ENTER)

                # clicar em calcular
                button = wait.until(EC.element_to_be_clickable((
                    By.CSS_SELECTOR,
                    "#card-calcular-precificacao > article > article > section > div > form > "
                    "div.col-xs-12.precificacao-content__calculate-button.col-no-padding > button"
                )))
                button.click()

                # aguarda tabela do fluxo
                wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "#card-fluxo-pagamento > article > article > section > div > div > table"
                )))

                # extrai VNE (VNA) e TaxaEmissao (Remuneração)
                vne, taxa_emissao = extrair_vne_e_taxa_emissao(wait)

                # parse da tabela com bs4
                soup = BeautifulSoup(driver.page_source, "html.parser")
                table = soup.select_one("#card-fluxo-pagamento > article > article > section > div > div > table")
                rows = table.find_all("tr")

                data_list = []
                for row in rows:
                    columns = row.find_all("td")
                    data = [col.text.strip() for col in columns]
                    if data:
                        data_list.append(data)

                df_append = pd.DataFrame(data_list)
                df_append["Ativo"] = ativo
                df_append["VNE"] = vne
                df_append["TaxaEmissao"] = taxa_emissao

                df = pd.concat([df, df_append], ignore_index=True)

            except Exception as e:
                print(f"Erro ao extrair/preencher taxa ANBIMA para {ativo}: {e}")

    # Definir os nomes das colunas (agora com VNE e TaxaEmissao)
    columns = [
        "Dados do evento",
        "Data de pagamento",
        "Prazos (dias úteis)",
        "Dias entre pagamentos",
        "Expectativa de juros (%)",
        "Juros projetados",
        "Amortizações",
        "Fluxo descontado (R$)",
        "Ativo",
        "VNE",
        "TaxaEmissao"
    ]

    # Se por algum motivo vier linha vazia/coluna faltando, evita crash hard:
    if df.shape[1] != len(columns):
        print(f"[WARN] Número de colunas capturadas = {df.shape[1]} | esperado = {len(columns)}")
        print("       Vou tentar ajustar mantendo o que foi capturado e adicionando colunas faltantes como NaN.")
        while df.shape[1] < len(columns):
            df[df.shape[1]] = None
        df = df.iloc[:, :len(columns)]

    df.columns = columns

    # Salvar o DataFrame em CSV
    df.to_csv("Dados/flux_deb.csv", index=False)
    print("Tabela salva com sucesso!")

except Exception as e:
    print(f"Ocorreu um erro: {e}")

finally:
    driver.quit()
