from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
import pandas as pd


def quantify_trend(df: pd.DataFrame):

    x = np.array(df.index).reshape(-1, 1)
    y = df['snr'].values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)

    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()

    return {'table': est2.summary(), 'slope': reg.coef_[0][0], 'int': reg.intercept_[0], 'p': est2.f_pvalue}


def value_count_percent_grouping(df: pd.DataFrame,
                                 colname: str,
                                 groupby: str):
    """ Counts values in specified column grouped by specified variable

        Parameters
        ----------
        df
            dataframe with data to analyze
        colname
            column name of data to analyze
            should specify a categorical variable unless you wanna have a bad time
        groupby
            column name used to group data

        Returns
        -------
        df_v
            output dataframe. columns = levels in groupby, rows = tally of colname data, as number of rows in df
        df_p
            output dataframe. columns = levels in groupby, rows = tally of colname data,
            as percent of number of rows in df
    """

    print(f"\nCounting {colname} values grouped by {groupby}...")

    grouped = df.groupby(groupby)
    df_v = pd.DataFrame()
    df_p = pd.DataFrame()
    for group in grouped.groups:
        g = grouped.get_group(group)
        n = g.shape[0]
        v = g[colname].value_counts()
        p = v * 100 / n

        df_v = pd.concat([df_v, pd.DataFrame(v)], axis=1)
        df_p = pd.concat([df_p, pd.DataFrame(p)], axis=1)

    df_v.columns = list(grouped.groups.keys())
    df_v.sort_index(inplace=True)
    df_v.index = [f"{colname}_{i}" for i in df_v.index]

    df_p.columns = list(grouped.groups.keys())
    df_p.sort_index(inplace=True)
    df_p.index = [f"{colname}_{i}" for i in df_p.index]

    return df_v, df_p


def generate_df_groupby_summary(df: pd.DataFrame,
                                groupby_column: str,
                                dv_column: str,
                                method: str,
                                missing_value: int or float or str or None = 0):
    """Summarizes data in given way using given grouping criteria.

        Parameters
        ----------
        df
            dataframe to group
        groupby_column
            column name in df that df is grouped by
        dv_column
            column name in df that is summarized
        method
            'count' or a stat in pd.DataFrame.describe() that is used to summarize.
        missing_value
            if using 'count' method and a category of dv_column is not found, it's assigned missing_value

        Returns
        -------
        dataframe of specified analysis
    """

    df_grouped = df.groupby(groupby_column)[dv_column]

    groups = df_grouped.groups.keys()

    if method == 'count':
        dv_values = sorted(df[dv_column].unique())
    if method in ['mean', '50%', 'std', 'min', 'max']:
        dv_values = [method]

    columns = [groupby_column] + dv_values

    data_out = []

    for group in groups:
        group_data = [group]
        g = df_grouped.get_group(group)

        if method == 'count':
            v = g.value_counts()

            for dv_val in dv_values:
                if dv_val in v.keys():
                    group_data.append(v.loc[dv_val])
                if dv_val not in v.keys():
                    group_data.append(missing_value)

        if method in ['mean', '50%', 'std', 'min', 'max']:
            d = g.describe()
            group_data.append(d[method])

        data_out.append(group_data)

    df_out = pd.DataFrame(data_out, columns=columns)

    return df_out
