import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
import torch


def feature_importance(learner, df):
    # define and split training/testing data and training the regressor
    features = df.iloc[:, 0:-1]
    target = df.iloc[:, -1]

    learner.fit(features, target)

    importance = learner.feature_importances_
    features = df.columns[0:-1]

    col = zip(features, importance)

    # sort and save features based on its importance into a dataframe
    importance_data = pd.DataFrame(col, columns=['feature', 'importance'])
    importance_data_asc = importance_data.sort_values('importance')

    # plot the importance data
    fig = plt.figure(figsize=(20, 15))
    plt.barh(y=importance_data_asc['feature'], width=importance_data_asc['importance'], height=0.9)

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.ylabel("Adsorbate Features", fontsize=15)
    plt.xlabel("Feature Importance", fontsize=15)
    plt.title("Feature Importance of Adsorbates", fontsize=15)
    plt.show();


def pearson_correlation(df, last=False):
    if last == False:
        data = df.iloc[:, :20]
    else:
        data = df.iloc[:, :-1]

    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=220),
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='right'
    )
    plt.rcParams['figure.figsize'] = (20, 20)


def trendline(xd, yd, i, name, rmse, ax, order=1, c='k', alpha=1, Rval=False):
    """Make a line of best fit"""

    # Calculate trendline
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order == 2 else 0

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept

    # Plot trendline
    if i <= 3:
        ax[0, i].plot(xl, yl, c, alpha=alpha, linestyle='--', linewidth=0.8)

    if i > 3:
        ax[1, i - 4].plot(xl, yl, c, alpha=alpha, linestyle='--', linewidth=0.8)

    # Calculate R Squared
    p = np.poly1d(coeffs)

    ybar = np.sum(yd) / len(yd)
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr = ssreg / sstot

    if not Rval:
        # Plot R^2 value
        if i <= 3:
            ax[0, i].text(-0.56, -0.80, name, fontsize=18)
            ax[0, i].text(-0.56, -0.85, '$RMSE = %0.4f$' % rmse, fontsize=18)
            ax[0, i].text(-0.56, -0.9, '$R^2 = %0.5f$' % Rsqr, fontsize=18)
        if i > 3:
            ax[1, i - 4].text(-0.56, -0.80, name, fontsize=18)
            ax[1, i - 4].text(-0.56, -0.85, '$RMSE = %0.4f$' % rmse, fontsize=18)
            ax[1, i - 4].text(-0.56, -0.9, '$R^2 = %0.5f$' % Rsqr, fontsize=18)
    else:
        # Return the R^2 value:
        return Rsqr


def plot_data(train_data, test_data, i, name, rmse, ax):
    if i <= 3:
        # Plot data
        ax[0, i].scatter(train_data['G_co (eV)'], train_data['prediction'], label="Training Set",
                         alpha=1, marker='o', s=25)
        ax[0, i].scatter(test_data['G_co (eV)'], test_data['prediction'], label='Testing Set',
                         alpha=1, marker='o', s=25)
        # ax[0,i].set_title(name, fontsize=15)

        # Combine data for best fit line
        x = pd.concat([train_data['G_co (eV)'], test_data['G_co (eV)']])
        y = pd.concat([train_data['prediction'], test_data['prediction']])

        # plot best fit line
        trendline(x, y, i, name, rmse, ax)

        # plot legend/axis
        ax[0, i].legend(loc='upper left', fontsize=15, frameon=False)

    if i > 3:
        # Plot data
        ax[1, i - 4].scatter(train_data['G_co (eV)'], train_data['prediction'], label="Training Set",
                             alpha=1, marker='o', s=25)
        ax[1, i - 4].scatter(test_data['G_co (eV)'], test_data['prediction'], label='Testing Set',
                             alpha=1, marker='o', s=25)
        # ax[1,i-4].set_title(name, fontsize=15)

        # Combine data to create best fit line
        x = pd.concat([train_data['G_co (eV)'], test_data['G_co (eV)']])
        y = pd.concat([train_data['prediction'], test_data['prediction']])

        # Plot best fit line
        trendline(x, y, i, name, rmse, ax)

        # Plot legend/axis
        ax[1, i - 4].legend(loc='upper left', fontsize=15, frameon=False)
def wakao():
    return

def transform_2D(series):
    result = []
    temp = []
    count = 1
    for i in series:
        if count % 18 != 0:
            temp.append(i)
            count += 1
        else:
            count = 1
            result.append(temp)
            temp = []
            temp.append(i)
            count += 1
    result.append(temp)
    return result

def seed_everything(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def set_axis_style(ax, labels):
    ax.get_yaxis().set_tick_params(direction='out')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels, rotation = 'horizontal')
    ax.set_ylim(0.25, len(labels) + 0.75)
