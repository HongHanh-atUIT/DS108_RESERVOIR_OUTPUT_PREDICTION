from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
from multiprocessing import Process

URL               = "https://www.evn.com.vn/c3/thong-tin-ho-thuy-dien/Muc-nuoc-cac-ho-thuy-dien-117-123.aspx"
TARGET_RESERVOIRS = ["Sông Ba Hạ", "Sông Tranh 2"]
HOURS             = [f"{h:02d}:00" for h in range(24)]

JS_SET_VALUE  = "arguments[0].value = arguments[1];"
JS_FIRE_EVENT = "arguments[0].dispatchEvent(new Event('change', {bubbles:true}));"


def open_web(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(4)
    return driver


def switch_to_iframe(driver):
    driver.switch_to.default_content()
    iframe = driver.find_element(By.TAG_NAME, "iframe")
    driver.switch_to.frame(iframe)
    time.sleep(0.3)


def select_reservoirs(driver):
    switch_to_iframe(driver)
    driver.find_element(By.CSS_SELECTOR, "button.multiselect.dropdown-toggle").click()
    time.sleep(1)
    for opt in driver.find_elements(By.CSS_SELECTOR, ".multiselect-container li"):
        label = opt.find_element(By.TAG_NAME, "label")
        cb    = opt.find_element(By.TAG_NAME, "input")
        want  = any(t in label.text for t in TARGET_RESERVOIRS)
        if cb.is_selected() != want:
            label.click()
            time.sleep(0.2)
    driver.find_element(By.CSS_SELECTOR, "button.multiselect.dropdown-toggle").click()
    time.sleep(0.5)


def query(driver, dt_str):
    switch_to_iframe(driver)
    inp = driver.find_element(By.ID, "UCViewHoChuaThuyDienPublic1_tbxDenNgay")
    # Truyền giá trị qua arguments[] thay vì nhúng vào f-string → tránh lỗi {
    driver.execute_script(JS_SET_VALUE, inp, dt_str)
    driver.execute_script(JS_FIRE_EVENT, inp)
    driver.find_element(By.CSS_SELECTOR, "a.btn-u4").click()
    time.sleep(0.1)
    return driver.page_source


def parse_table(html, date, hour):
    soup  = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="tblgridtd")
    if not table:
        return []

    def to_float(cells, i):
        try:
            return float(cells[i].get_text(strip=True).replace(",", "."))
        except:
            return None

    records = []
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 9:
            continue
        b = cells[0].find("b")
        if not b or not any(t in b.text for t in TARGET_RESERVOIRS):
            continue
        records.append({
            "ten_ho":   b.text.strip(),
            "ngay":     date.strftime("%Y-%m-%d"),
            "gio":      hour,
            "H_tl":     to_float(cells, 2),
            "H_dbt":    to_float(cells, 3),
            "H_c":      to_float(cells, 4),
            "Q_ve":     to_float(cells, 5),
            "Sigma_Qx": to_float(cells, 6),
            "Q_xt":     to_float(cells, 7),
            "Q_xm":     to_float(cells, 8),
            "N_cxs":    to_float(cells, 9)  if len(cells) > 9  else None,
            "N_cxm":    to_float(cells, 10) if len(cells) > 10 else None,
        })
    return records


def scrape_year(year):
    print(f"[{year}] Bắt đầu...")
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)

    driver = open_web(URL)
    select_reservoirs(driver)

    records = []
    day = start
    while day <= end:
        print(f"  [{year}] {day.strftime('%d/%m/%Y')} ...", end=" ", flush=True)
        for hour in HOURS:
            dt_str = day.strftime("%d/%m/%Y") + f" {hour}"
            html   = query(driver, dt_str)
            records.extend(parse_table(html, day, hour))
        day += timedelta(days=1)

    driver.quit()

    fname = f"evn_mucnuoc_{year}.csv"
    pd.DataFrame(records).to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"[{year}] Đã lưu: {fname} ({len(records)} dòng)")


if __name__ == "__main__":
    years = [2022, 2023, 2024, 2025]

    processes = []
    for year in years:
        p = Process(target=scrape_year, args=(year,))
        p.start()
        processes.append(p)
        time.sleep(2)

    for p in processes:
        p.join()

    print("\nTất cả năm đã hoàn thành!")