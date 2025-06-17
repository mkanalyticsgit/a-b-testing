import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# Random seed for reproducibility
np.random.seed(42)
# Parameters
n = 3000  # Users per group
baseline_mean = 48  # Average watch time in control
effect = 3.5  # Hypothetical increase in minutes with chat overlay
std_dev = 12
# Simulate data
control = np.random.normal(loc=baseline_mean, scale=std_dev, size=n)
treatment = np.random.normal(loc=baseline_mean + effect, scale=std_dev, size=n)
# Combine into DataFrame
df = pd.DataFrame({
   'group': ['control'] * n + ['treatment'] * n,
   'watch_time': np.concatenate([control, treatment])
})
# Summary stats
summary = df.groupby('group')['watch_time'].agg(['mean', 'std', 'count'])
print("Summary Statistics:\n", summary)
# Visualize
plt.figure(figsize=(10,5))
plt.hist(df[df['group'] == 'control']['watch_time'], bins=40, alpha=0.6, label='Control', color='blue')
plt.hist(df[df['group'] == 'treatment']['watch_time'], bins=40, alpha=0.6, label='Treatment', color='green')
plt.title('Watch Time Distribution - Chat Overlay A/B Test')
plt.xlabel('Watch Time (min)')
plt.ylabel('Users')
plt.legend()
plt.show()
# T-test
t_stat, p_val = stats.ttest_ind(
   df[df['group'] == 'control']['watch_time'],
   df[df['group'] == 'treatment']['watch_time'],
   equal_var=False
)
print(f"\nT-test: t = {t_stat:.3f}, p = {p_val:.4f}")
if p_val < 0.05:
   print("✅ Statistically significant difference — chat overlay improved watch time.")
else:
   print("❌ No statistically significant difference detected.")
