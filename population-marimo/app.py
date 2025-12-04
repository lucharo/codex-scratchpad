import marimo
__generated_with = "0.14.13"
app = marimo.App()
import marimo as mo
import polars as pl
import numpy as np
from pathlib import Path

__all__ = ["app"]

@app.cell
def setup():
    DATA_DIR = Path(__file__).parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

    @mo.persistent_cache()
    def fetch_indicator(indicator: str) -> pl.DataFrame:
        """Load indicator from CSV or attempt download."""
        path = DATA_DIR / f"{indicator}.csv"
        if path.exists():
            return pl.read_csv(path)
        try:
            import pandas as pd, requests, io, zipfile
            url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?downloadformat=csv"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
                with zf.open(csv_name) as f:
                    pdf = pd.read_csv(f, skiprows=4)
            pdf = pdf[["Country Name", "2020"]].rename(columns={"Country Name": "Country", "2020": indicator})
            df = pl.from_pandas(pdf.dropna())
            df.write_csv(path)
            return df
        except Exception:
            return pl.DataFrame()

    return fetch_indicator, DATA_DIR

@app.cell
def load_data(setup):
    fetch_indicator, DATA_DIR = setup
    indicators = {
        "birth_rate": "SP.DYN.CBRT.IN",
        "death_rate": "SP.DYN.CDRT.IN",
        "life_expectancy": "SP.DYN.LE00.IN",
        "fertility": "SP.DYN.TFRT.IN",
    }
    wb = {name: fetch_indicator(code) for name, code in indicators.items()}
    return wb

@app.cell
def data_dictionary():
    table = pl.DataFrame(
        {
            "Indicator": [
                "Birth rate (per 1000)",
                "Death rate (per 1000)",
                "Life expectancy at birth",
                "Total fertility rate",
            ],
            "Code": [
                "SP.DYN.CBRT.IN",
                "SP.DYN.CDRT.IN",
                "SP.DYN.LE00.IN",
                "SP.DYN.TFRT.IN",
            ],
        }
    )
    table
    return table

@app.cell
def header(mo):
    mo.md(
        """# Population Scenario Explorer

Adjust the controls below to modulate global life expectancy and natality.
Data come from the World Bank API (cached under `data/`)."""
    )
    return

@app.cell
def ui(mo):
    le_delta = mo.ui.slider(-10, 30, 0, label="Life expectancy delta (yrs)")
    natality = mo.ui.slider(0.2, 1.2, 1.0, step=0.05, label="Natality multiplier")
    capacity = mo.ui.number(10_000_000_000, label="Planet carrying capacity")
    end_year = mo.ui.slider(2030, 2100, 2050, step=10, label="Projection end year")
    mo.md(f"{le_delta} {natality} {capacity} {end_year}")
    return le_delta, natality, capacity, end_year

@app.cell
def model(load_data, ui, mo):
    le_delta, natality, capacity, end_year = ui
    birth = load_data["birth_rate"].rename({"SP.DYN.CBRT.IN": "birth"})
    death = load_data["death_rate"].rename({"SP.DYN.CDRT.IN": "death"})
    life = load_data["life_expectancy"].rename({"SP.DYN.LE00.IN": "life"})
    combined = birth.join(death, on="Country", how="inner").join(life, on="Country")
    combined = combined.drop_nulls()
    if combined.is_empty():
        mo.stop("No data available. Please add CSVs to data/.")
    avg_birth = combined["birth"].mean()
    avg_death = combined["death"].mean()
    avg_life = combined["life"].mean()
    adj_death = avg_death * (avg_life / (avg_life + le_delta.value))
    r = (avg_birth * natality.value - adj_death) / 1000
    years = np.arange(2020, end_year.value + 1)
    pop = np.zeros(len(years))
    pop[0] = 8_000_000_000
    for i in range(1, len(years)):
        growth = r * pop[i-1] * (1 - pop[i-1] / capacity.value)
        pop[i] = pop[i-1] + growth
    df = pl.DataFrame({"Year": years, "Population": pop})
    return df

@app.cell
def chart(model, ui):
    _, _, capacity, _ = ui
    cap_line = pl.DataFrame({"Year": model["Year"], "Capacity": capacity.value})
    chart = (
        alt.Chart(model.to_pandas())
        .mark_line()
        .encode(x="Year", y="Population")
        + alt.Chart(cap_line.to_pandas())
        .mark_line(color="red", strokeDash=[4,4])
        .encode(x="Year", y="Capacity")
    )
    chart
    return chart

@app.cell
def references(mo):
    mo.md(
        """## References
- UN World Population Prospects (TODO)
- World Bank Open Data
- TODO: carrying capacity studies"""
    )
    return

if __name__ == "__main__":
    app.run()
