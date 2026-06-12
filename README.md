# Drug Utilization Review (DUR) + Visual Drug-Drug interaction Dashboard

Tool that utlizies the OnSIDES database: Extracting adverse drug events from drug labels using natural language processing models (https://pubmed.ncbi.nlm.nih.gov/40179876/) and utilizies RxNorm mapping and adverse-event analytics to highlight the most frequent interactions as percentage-based gauges—transforming complex DUR data into actionable clinical insight for pharmacists


![DUR Dashboard](dur_one.jpg)

# Easy usage guide in Kaggle

`!git clone "https://github.com/asvcode/DUR.git"`

Ensure working in the correct directory:

`cd /kaggle/working/DUR`

`pip install -e`

For full functionality of LLM use - requires an OpenAI account and use `UserSecretsClient()` to store your passkeys

```
user_secrets = UserSecretsClient()
my_secret_value = user_secrets.get_secret("openai_kaggle") 
os.environ["OPENAI_API_KEY"] = my_secret_value
client = OpenAI()
```

Run the Python script located at:

`!python /kaggle/working/DUR/src/dur_mvp/utils.py`

For DUR list and dashboard:

```
from src.dur_mvp.utils import get_dur
grade, table = get_dur(
    "escitalopram", "ondansetron",
    use_curated_for_high_risk=False,
    combine_method="max",
    top_n=7,
    llm_mode="blend", #off, always, blend
    use_api=True,       # live LLM
    df_pt=df_pt,
    # config="config/dur.yaml",  # if you adopted config-driven review
)
```






