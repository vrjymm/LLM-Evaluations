import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load Data
df = pd.read_csv("summary_eval_scores.csv")

# Extract available metrics with 'Score'
score_columns = [col for col in df.columns if "Score" in col and col != "Testcase ID"]
models = df["Model"].unique()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Summary Evaluation Dashboard"

app.layout = html.Div([
    html.H1("📊 Summary Evaluation Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id="metric-dropdown",
            options=[{"label": col, "value": col} for col in score_columns],
            value=score_columns[0]
        ),
        html.Label("Select Model(s):"),
        dcc.Checklist(
            id="model-checklist",
            options=[{"label": model, "value": model} for model in models],
            value=list(models),
            inline=True
        )
    ], style={"width": "80%", "margin": "auto"}),

    dcc.Graph(id="metric-bar-chart"),

    html.Footer("Created by Vrinda • GenAI Evaluation", style={"textAlign": "center", "marginTop": "2rem"})
])


@app.callback(
    Output("metric-bar-chart", "figure"),
    [Input("metric-dropdown", "value"),
     Input("model-checklist", "value")]
)
def update_bar_chart(selected_metric, selected_models):
    filtered_df = df[df["Model"].isin(selected_models)]
    fig = px.bar(
        filtered_df,
        x="Testcase ID",
        y=selected_metric,
        color="Model",
        barmode="group",
        title=f"{selected_metric} Across Models and Testcases",
        height=500
    )
    fig.update_layout(legend_title_text="Model")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
