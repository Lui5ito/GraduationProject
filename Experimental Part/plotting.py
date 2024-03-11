from packages import *

def eval_reg(true, pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axis
    else:
        plt.sca(ax) 

    sns.set_theme(context='paper', font_scale=1.5)
    sns.scatterplot(x=true, y=pred, color='blue', alpha=0.5)

    ## Adding the x=y line and the text
    plt.plot([min(true), max(true)], [min(true), max(true)], linestyle='--', color='red', linewidth=2, alpha = 0.8)

    plt.title('Predicted vs. Actual Values of Y')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    if ax is None:
        plt.show()

def eval_multireg(true, pred):
    fig, axes = plt.subplots(1, true.shape[1], figsize=(6*true.shape[1], 5))

    for i, ax in enumerate(axes):
        eval_reg(true[:, i], pred[:, i], axes[i])
        ax.set_title(f'Output {i+1}')

    plt.tight_layout()
    plt.show()