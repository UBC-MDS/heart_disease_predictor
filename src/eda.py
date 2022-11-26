# author:
# date:

"""
A utility script to perform exploratory data analysis on the dataset used in ....

Usage: eda.py [--from=<raw_file_path>] [--to=<processed_dir_path>]
Options:
[--from=<path to training data >]     Path to the training dataset csv file

[--to=<directory to save results>]    Directory path to save the EDA artifacts
                            
Uses the docopt for command-line argument parsing (add other packages used)
- http://docopt.org/
"""

from docopt import docopt
import os
import pandas as pd
import altair as alt
import vl_convert as vlc

alt.renderers.enable("mimetype")

opt = docopt(__doc__)

default_from = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, "results")


def eda(from_path, to_path):
    if from_path is None:
        from_path = default_from
    if to_path is None:
        to_path = default_to
    print("\nPerforming EDA on the dataset: to be implemented ...")
    print(f"Reading data from {from_path}")

    # read train_df.csv
    train_df = pd.read_csv(from_path + "/train_heart.csv")
    test_df = pd.read_csv(from_path + "/test_heart.csv")
    # create split count talbe
    class_count = pd.concat(
        [
            train_df["target"]
            .value_counts()
            .rename({1: "Presence of HD:", 0: "No presence of HD:"}),
            test_df["target"]
            .value_counts()
            .rename({1: "Presence of HD:", 0: "No presence of HD:"}),
        ],
        axis=1,
    )
    class_count.columns = ["Training", "Test"]
    class_count.T.to_csv(f"{to_path}/class_count.csv")

    # Spearman's correlation values for all of the features and target value
    cor_data = (
        train_df.select_dtypes("number")
        .corr()
        .stack()
        .reset_index()
        .rename(
            columns={0: "correlation", "level_0": "variable", "level_1": "variable2"}
        )
    )
    cor_data["coefficient"] = cor_data["correlation"].map("{:.3f}".format)

    base = alt.Chart(cor_data).encode(x="variable2:O", y="variable:O")

    text = base.mark_text().encode(
        text="coefficient",
    )

    corr_plot = base.mark_rect().encode(
        color=alt.Color(
            "correlation:Q", scale=alt.Scale(domain=(-1, 1), scheme="purpleorange")
        )
    )

    combined_corr = (corr_plot + text).properties(height=600, width=600)
    # Analysis of numeric feature distributions
    numeric_cols = [
        "age",
        "resting_blood_pressure",
        "cholesterol",
        "max_hr_achieved",
        "oldpeak",
    ]
    cate_cols = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "resting_ecg_results",
        "exercise_induced_angina",
        "slope",
        "num_major_vessels",
        "thalassemia",
    ]

    numeric_dist = (
        alt.Chart(train_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=30)),
            y="count()",
            color="target:N",
        )
        .properties(height=200, width=200)
        .repeat(numeric_cols, columns=3)
    )
    numeric_dist = numeric_dist.properties(title="Distribution of Numeric Features")

    # Analysis of categorical feature distributions
    cate_dist = (
        alt.Chart(train_df)
        .mark_bar(opacity=0.8)
        .encode(x=alt.X(alt.repeat(), type="nominal"), y="count()", color="target:N")
        .properties(height=200, width=200)
        .repeat(cate_cols, columns=3)
    )
    cate_dist = cate_dist.properties(title="Distribution of Categorical Features")

    # `thalach` vs `oldpeak` for the two target classes
    thalach_age_plot = (
        alt.Chart(train_df, title="Maximum heart rate vs. Age")
        .mark_circle()
        .encode(
            x=alt.X("age", scale=alt.Scale(zero=False), title="Age"),
            y=alt.Y(
                "max_hr_achieved",
                scale=alt.Scale(zero=False),
                title="Maximum heart rate",
            ),
            color=alt.Color("target:N", title="Presence of HD"),
        )
    )
    combined_thalach = thalach_age_plot + thalach_age_plot.mark_line(
        size=3
    ).transform_regression("age", "max_hr_achieved", groupby=["target"])

    # Plotting the correlations
    corr_viz = (
        alt.Chart(train_df)
        .mark_circle(opacity=0.8, size=10)
        .encode(
            x=alt.X(
                alt.repeat("row"), type="quantitative", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(alt.repeat("column"), type="quantitative"),
            color=alt.Y("target:N"),
        )
        .properties(height=170, width=170)
        .repeat(column=numeric_cols, row=numeric_cols)
    )
    corr_viz = corr_viz.properties(title="Scatter Plot of Numeric Feature Pairs")

    # function provided by Dr. Joel Ostblom
    def save_chart(chart, filename, scale_factor=1):
        """
        Save an Altair chart using vl-convert

        Parameters
        ----------
        chart : altair.Chart
            Altair chart to save
        filename : str
            The path to save the chart to
        scale_factor: int or float
            The factor to scale the image resolution by.
            E.g. A value of `2` means two times the default resolution.
        """
        if filename.split(".")[-1] == "svg":
            with open(filename, "w") as f:
                f.write(vlc.vegalite_to_svg(chart.to_dict()))
        elif filename.split(".")[-1] == "png":
            with open(filename, "wb") as f:
                f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
        else:
            raise ValueError("Only svg and png formats are supported")

    plot_list = [combined_corr, numeric_dist, cate_dist, combined_thalach, corr_viz]
    plot_names = [
        "correlation_matrix",
        "numeric_distributions",
        "categorical_distributions",
        "thalach_vs_age",
        "correlation_scatter",
    ]
    for plot, name in zip(plot_list, plot_names):
        plot_name = f"{to_path}/{name}.png"
        print(f"saving: {name}")
        save_chart(plot, plot_name, 2)

    print(f"Saving EDA artifacts to {to_path}")


def main(from_path, to_path):
    eda(from_path, to_path)


if __name__ == "__main__":
    main(opt["--from"], opt["--to"])
