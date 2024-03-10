from packages import *

def eval_reg(true, pred):
    plt.figure()

    sns.set_theme(context='paper', font_scale=1.5)
    sns.scatterplot(x=true, y=pred, color='blue', alpha=0.5)

    ## Adding the x=y line and the text
    plt.plot([min(true), max(true)], [min(true), max(true)], linestyle='--', color='red', linewidth=2, alpha = 0.8)

    plt.title('Predicted vs. Actual Values of Y')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.show()