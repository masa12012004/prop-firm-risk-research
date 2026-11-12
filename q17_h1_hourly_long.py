"""Save the hour-by-hour long-history H1 hybrid position (computed by q16, unchanged) for the challenge simulator."""
import os, pandas as pd
import q16_gold_h1_long_history as q
out = pd.DataFrame({"close": q.h1["close"], "ret": q.h1["ret"], "pos": q.hyb, "cubic": q.cubic, "regime": q.regime,
                    "tday": q.h1["tday"]})
out.to_parquet(os.path.join(q.RES, "q17_h1_hourly_long.parquet"))
print("saved", len(out))
