import json
import pandas as pd
from typing import List, Dict, Optional
from tabulate import tabulate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    SummarizationMetric, ToxicityMetric, BiasMetric,
    PromptAlignmentMetric, HallucinationMetric,
    JsonCorrectnessMetric, GEval
)
from rouge_score import rouge_scorer
import bert_score
from anthropic import Anthropic


class CustomClaudeSonnet(DeepEvalBaseLLM):
    def __init__(self):
        self.model = Anthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
        )
        return "".join([block.text for block in resp.content])

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Claude-3 Sonnet"


class Metric:
    def __init__(self, model=None, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        raise NotImplementedError


class RougeMetric(Metric):
    def measure(self, reference: str, summary: str) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, summary)
        return {k: v.fmeasure for k, v in scores.items()}


class BertScoreMetric(Metric):
    def measure(self, reference: str, summary: str) -> float:
        _, _, f1 = bert_score.score([summary], [reference], lang="en", verbose=False)
        return f1[0].item()


class DeepEvalMetricWrapper(Metric):
    def __init__(self, model, metric_cls, **kwargs):
        super().__init__(model=model, threshold=kwargs.get('threshold', 0.5))
        self.metric_cls = metric_cls
        self.metric_kwargs = kwargs

    def measure(self, test_case: LLMTestCase) -> Dict[str, str]:
        metric = self.metric_cls(threshold=self.threshold, model=self.model, **self.metric_kwargs)
        metric.measure(test_case)
        return {"score": metric.score, "reason": metric.reason}


class GEvalMetric(Metric):
    def __init__(self, model, definition: dict, threshold=0.5):
        super().__init__(model, threshold)
        self.definition = definition

    def create(self) -> GEval:
        params = [
            param for name, param in LLMTestCaseParams.__dict__.items()
            if not name.startswith('__') and self.definition['params'].get(name.lower())
        ]
        return GEval(
            name=self.definition['name'],
            criteria=self.definition['criterion'],
            evaluation_steps=self.definition['steps'],
            evaluation_params=params,
            model=self.model,
            threshold=self.threshold
        )

    def measure(self, test_case: LLMTestCase) -> Dict[str, str]:
        metric = self.create()
        metric.measure(test_case)
        return {"score": metric.score, "reason": metric.reason}


def generate_test_case(input_data: dict, models: List[str]) -> Dict[str, LLMTestCase]:
    return {
        model: LLMTestCase(
            input=input_data['source'],
            actual_output=input_data[f"{model}_summary"],
            expected_output=input_data.get('reference_summary'), 
            context = [input_data['source']]
        ) for model in models
    }


def load_dataset_to_dataframe(json_path: str, models: List[str]) -> pd.DataFrame:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    records = []
    for idx, item in enumerate(raw_data):
        row = {
            "testcase_id": idx,
            "source": item["source"],
            "reference_summary": item.get("reference_summary", "")
        }
        for model in models:
            row[f"{model}_summary"] = item.get(f"{model}_summary", "")
        records.append(row)

    return pd.DataFrame(records)



def read_json_dataset(json_file_path: str, models: List[str]) -> Dict:
    with open(json_file_path, "r") as file:
        data = json.load(file)

    dataset = {model: [] for model in models}
    for item in data:
        cases = generate_test_case(item, models)
        for model in models:
            dataset[model].append(cases[model])

    return {
        'dataset': dataset,
        'models': models,
        'num_testcases': len(data)
    }


def evaluate_all_metrics(test_case: LLMTestCase, model=None) -> Dict:
    correctness_metric_def = {
    'name': "Correctness",
    'criterion': "Determine whether the actual output is factually correct based on the expected output.",
    'steps': [
    "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
    "You should also heavily penalize omission of detail",
    "Vague language, or contradicting OPINIONS, are OK"
    ],
    'params': {'input': True, 'actual_output': True, 'expected_output': True, 'context': False}
    }

    rouge = RougeMetric().measure(test_case.expected_output, test_case.actual_output)
    bert = BertScoreMetric().measure(test_case.expected_output, test_case.actual_output)

    results = {
    "ROUGE Scores": rouge,
    "BERTScore": bert
    }

    if model:
        results.update({
            "Bias": DeepEvalMetricWrapper(model, BiasMetric).measure(test_case),
            "Toxicity": DeepEvalMetricWrapper(model, ToxicityMetric).measure(test_case),
            "Summarization": DeepEvalMetricWrapper(model, SummarizationMetric).measure(test_case),
            "Prompt Alignment": DeepEvalMetricWrapper(
                model, PromptAlignmentMetric, prompt_instructions=["Reply in all uppercase"]
            ).measure(test_case),
            "Hallucination": DeepEvalMetricWrapper(
                model, HallucinationMetric
            ).measure(test_case),
            "GEval Correctness": GEvalMetric(model, correctness_metric_def).measure(test_case)
        })

    return results


def evaluate_and_format_results(df: pd.DataFrame, models: List[str], judge_model) -> pd.DataFrame:
    all_results = []

    for _, row in df.iterrows():
        for model in models:
            test_case = LLMTestCase(
                input=row["source"],
                actual_output=row[f"{model}_summary"],
                expected_output=row["reference_summary"],
                context=[row["source"]]
            )
            results = evaluate_all_metrics(test_case, model=judge_model)
            flat_result = {
                "Testcase ID": row["testcase_id"],
                "Model": model,
                **flatten_dict(results)
            }
            all_results.append(flat_result)

    results_df = pd.DataFrame(all_results)
    return results_df

def flatten_dict(d: Dict, parent_key='', sep=' - ') -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            if 'score' in v and 'reason' in v:
                items.append((f"{k} Score", v['score']))
                items.append((f"{k} Reason", v['reason']))
            else:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    dataset_file = "conversation_summarization_dataset.json"
    test_models = ['model_1', 'model_2', 'model_3']
    judge_model = CustomClaudeSonnet()

    df = load_dataset_to_dataframe(dataset_file, test_models)
    results_df = evaluate_and_format_results(df, test_models, judge_model)

    # Display nicely in terminal
    print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    # Optional: save results
    results_df.to_csv("summary_eval_scores.csv", index=False)
    print("\n✅ Results saved to summary_eval_scores.csv")


if __name__ == "__main__":
    main()

    
