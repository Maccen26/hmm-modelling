# Humidity cleaning: relative humidity → moisture excess

Source: `drivers/clean_dtu_data.py` (`calibrate_humidity` and helpers).

## Why not use relative humidity directly?

Relative humidity (RH, %) depends on temperature as well as on how much water is in the air:
warming a room lowers its RH even when no moisture is added or removed. Occupants add moisture
by breathing, much as they add CO₂, so the useful signal is the *absolute* amount of water
vapour. On top of that, the indoor level follows the outdoor level through ventilation. So we
convert RH to a **humidity ratio** (g water per kg dry air) and subtract the outdoor humidity ratio.

## Step 1: Saturation vapour pressure (Magnus formula)

The Magnus formula over water (Sonntag 1990), with temperature $T$ in °C, gives

$$
e_s(T) = 6.112 \exp\!\left(\frac{17.62\,T}{243.12 + T}\right) \quad [\text{hPa}],
$$

valid for roughly $-45$ to $60$ °C (`saturation_vapour_pressure`).

## Step 2: Actual vapour pressure

RH is the ratio of actual to saturation vapour pressure, so

$$
e = \frac{\mathrm{RH}}{100}\, e_s(T) \quad [\text{hPa}]
$$

(`vapour_pressure`).

## Step 3: Humidity ratio

The mass of water vapour per mass of dry air, at total pressure $p$, is

$$
x = 622 \cdot \frac{e}{p - e} \quad [\text{g/kg dry air}],
$$

where $622 \approx 1000 \cdot M_{\text{water}}/M_{\text{dry air}} = 1000 \cdot 18.015/28.964$.
We fix $p = 1013$ hPa (standard atmosphere, `P_ATM_HPA`) because the room sensors don't
measure pressure. Real pressure varies by about ±3%, and the error in $x$ is about the same
relative size (`humidity_ratio`).

## Step 4: Indoor and outdoor ratios

- **Indoor** $x_\text{in}$: the room's own `humidity` (RH) and `temperature` columns, after
  they have been resampled to the 30 min grid and cut to the longest gap-free segment.
- **Outdoor** $x_\text{out}$: hourly `mean_relative_hum` and `mean_temp` from DMI's 10 km
  climate grid cell covering DTU (downloaded by `drivers/fetch_humidity.py` to
  `raw/dtu/humidity.csv`). We push them through the same formulas.

## Step 5: Aligning the two time axes

Room timestamps are local and tz-naive; DMI timestamps (`DateTo`, interval end) are UTC. For each bin:

1. Localise to `Europe/Copenhagen`. At the autumn DST switch, the ambiguous hour becomes `NaT`.
   At the spring switch, nonexistent times are shifted forward.
2. Convert to UTC.
3. Check that the segment lies inside the DMI data's span (±1 h tolerance), and raise an error
   otherwise. We never extrapolate.
4. Match each bin to the **nearest** hourly DMI value (`pd.merge_asof(direction="nearest")`).
5. Bins dropped as ambiguous at DST get forward/back-filled from their neighbours.

## Step 6: Moisture excess

The `humidity` column is overwritten with

$$
\Delta x_t = x_{\text{in},t} - x_{\text{out},t} \quad [\text{g/kg dry air}].
$$

$\Delta x \approx 0$ means the room is at outdoor moisture (empty or well ventilated). Positive
values are moisture added indoors, mainly by occupants. This makes it the humidity counterpart
of the CO₂ baseline calibration.

## Bookkeeping

- `temperature` is loaded only to compute $x_\text{in}$, and is dropped afterwards unless it was
  requested in `y_cols`.
- $x_\text{out}$ is saved as `x_out_{train,val,test}.csv` so the transform can be partly
  undone ($x_\text{in} = \Delta x + x_\text{out}$). Getting back to RH would also need the
  indoor temperature.
- `metadata.csv` records `humidity_unit = "g/kg (x_in - x_out)"`.
- The plausibility filter `0 ≤ RH ≤ 100` is applied to the **raw** readings, before
  aggregation and conversion.


$$
d\tilde{C_t} = P - \alpha \tilde{C_t} \, dt + \sigma \, dW_t
$$
