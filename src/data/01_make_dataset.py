##### Credit Score Model #####

##### 0. Imports #####

### Data manipulation 
import pandas                   as pd
import numpy                    as np

### Data visualization
import seaborn                  as sns
import matplotlib               as mpl
import matplotlib.pyplot        as plt

### Statistics and Machine learning 
from sklearn.metrics            import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import RobustScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier

### Other libraries
import inflection
import warnings
import locale
import lxml

##### 1. Settings #####

### Ignoring warnings
warnings.filterwarnings('ignore')

### Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

### Visualization Settings
mpl.style.use('ggplot')

mpl.rcParams['figure.titlesize']    = 24
mpl.rcParams['figure.figsize']      = (20, 5)
mpl.rcParams['axes.facecolor']      = 'white'
mpl.rcParams['axes.linewidth']      = 1
mpl.rcParams['xtick.color']         = 'black'
mpl.rcParams['ytick.color']         = 'black'
mpl.rcParams['grid.color']          = 'lightgray'
mpl.rcParams['figure.dpi']          = 150
mpl.rcParams['axes.grid']           = True
mpl.rcParams['font.size']           = 12

sns.set_palette('rocket')

### Set the locale to the United States
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8');

##### 2. Functions #####

def set_image(title, xlabel, ylabel, rotation=0):
    """
    Summary: This function sets the image configuration.

    Args:
        title: the title of the plot.
        xlabel: the label for the x axis.
        ylabel: the label for the y axis.
        rotation: the rotation of the labels. default as 0.

    Returns: None
    """

    plt.title(title)
    plt.xlabel(xlabel, color='black')
    plt.ylabel(ylabel, color='black')
    plt.xticks(rotation=rotation)
    plt.tick_params(left=False, bottom=False);

    return None

##### 3. Loading Data #####

# Importing data as a dataframe 
df_raw = pd.read_csv('../../data/raw/raw_data.csv')

# Copying the dataframe to work with 'df'
df = df_raw.copy()

##### 4. Feature Engineering #####

# Column ID has no relevant information
df = df.drop(columns={'ID'})

# Renaming columns to snake_case style
old_columns = df.columns.tolist()
snake_case = lambda x: inflection.underscore(x)
new_columns = list(map(snake_case, old_columns))
df.columns = new_columns

# Converting types 

# Sum values
df['tl_sum']            = df['tl_sum'].str.replace('$', '').str.replace(',', '').astype(float)
df['tl_max_sum']        = df['tl_max_sum'].str.replace('$', '').str.replace(',', '').astype(float)

# Percentage values
df['tl_bal_hc_pct']     = df['tl_bal_hc_pct'].str.rstrip('% ').astype(float) / 100
df['tl_sat_pct']        = df['tl_sat_pct'].str.rstrip('% ').astype(float) / 100
df['tl_open_pct']       = df['tl_open_pct'].str.rstrip('% ').astype(float) / 100
df['tl_open24_pct']     = df['tl_open24_pct'].str.rstrip('% ').astype(float) / 100

# Filling missing values
df['inq_time_last'].fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)

# Exporting Dataframe to csv
df.to_csv('../../data/processed/processed_data.csv', index=False)

##### 5. Machine Learning #####

# Separating features and target
X = df.iloc[:, 1:28].values
y = df.iloc[:, 0].values

# Training and testing at 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling model
scaler      = RobustScaler()

# Fit the scaler to data
scaler.fit(X_train)

# Transform both the training and test data using the same scaler
X_train     = scaler.transform(X_train)
X_test      = scaler.transform(X_test)

# Logistic Regression
classifier  = LogisticRegression()

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting
y_pred = classifier.predict(X_test)

# Saving Predictions
predictions = classifier.predict_proba(X_test)

# Creating a dataframe with probabilities
df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])

# Adding the predicted target
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['pred_target'])

# Adding the actual outcome
df_test_dataset = pd.DataFrame(y_test, columns = ['actual_outcome'])

# Concatenating
output = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis = 1)

# Sorting by probability of non-default
output.sort_values(by='prob_0', ascending = False, inplace = True)

# Crating deciles
output['decile'] = pd.qcut(output['prob_0'], q=10, labels=False)

# Deciles in inversed order
output['decile'] = (9 - output['decile']) + 1

# Exporting Dataframe to csv
output.to_csv('../../data/processed/classifier_output.csv', index=False)

##### 6. Business Performance #####

# Empty dictionary
bp = {
    'decile':[],
    'count_of_decile': [],
    'sum_of_actual_outcome': [],
    'min_prob_good': [],
    'good': []
}

# Loop through each decile 
for i in range(1,11):
    aux = output.loc[output['decile'] == i]
    
    count_of_decile = len(aux)
    sum_of_actual_outcome = aux['actual_outcome'].sum()
    min_prob_good = aux['prob_0'].min()
    good = count_of_decile - sum_of_actual_outcome

    bp['decile'].append(i)
    bp['count_of_decile'].append(count_of_decile)
    bp['sum_of_actual_outcome'].append(sum_of_actual_outcome)
    bp['min_prob_good'].append(min_prob_good)
    bp['good'].append(good)

# From dictionary to dataframe
bp = pd.DataFrame.from_dict(bp)

# Adding cumulative values
for row in bp.index:
    bp.loc[row, 'cumm_good']    = bp.head(row+1)['good'].sum()
    bp.loc[row, 'cumm_bad']     = bp.head(row+1)['sum_of_actual_outcome'].sum()

# Adding percentages
bp['cumm_good_perc']            = bp['cumm_good']/bp['good'].sum()
bp['cumm_bad_perc']             = bp['cumm_bad']/bp['sum_of_actual_outcome'].sum()
bp['cumm_bad_avoided_perc']     = 1 - bp['cumm_bad_perc']

# Exporting Dataframe to csv
bp.to_csv('../../data/processed/df_metrics.csv', index=False)

# Copy
business_results = bp.copy()

# Changing data types
business_results['cumm_good']               = business_results['cumm_good'].astype(int)
business_results['cumm_bad']                = business_results['cumm_bad'].astype(int)

# Formatting
business_results['min_prob_good']           = (np.round(business_results['min_prob_good'] * 100, 1)).astype(str) + '%'
business_results['cumm_good_perc']          = (np.round(business_results['cumm_good_perc'] * 100, 1)).astype(str) + '%'
business_results['cumm_bad_perc']           = (np.round(business_results['cumm_bad_perc'] * 100, 1)).astype(str) + '%'
business_results['cumm_bad_avoided_perc']   = (np.round(business_results['cumm_bad_avoided_perc'] * 100, 1)).astype(str) + '%'

# Calculating the profit to business
business_results['profit_to_business']      = business_results['cumm_good'] * 100 - business_results['cumm_bad'] * 500

# Apply locale formatting to the column
business_results['profit_to_business']      = business_results['profit_to_business'].apply(lambda x: locale.format_string("%d", x, grouping=True))

# Formatting the column as currency
business_results['profit_to_business']      = '$' + ((business_results['profit_to_business']).astype(str))

# Exporting Dataframe to csv
business_results.to_csv('../../data/processed/df_results.csv', index=False)

##### 7. Figures #####

### Figure 01 Heatmap

# Increase the size of the plot
plt.figure(figsize=(20, 18))

# Correlation
fig01 = sns.heatmap(df.corr(), annot=True, fmt='.2f')

fig01 = set_image('Correlation Heatmap', None, None, 90)

plt.savefig('../../images/heatmap.png', dpi=150, format='png', bbox_inches='tight')

### Figure 02 Default Status

fg02 = sns.countplot(df, x='target', saturation=0.5)

fig02 = set_image('Target Variable Distribution', 'Default Status', 'Total')

plt.savefig('../../images/default_status.png', dpi=150, format='png', bbox_inches='tight')

### Figure 03 Boxplot 1

fig03 = sns.boxplot(data=df, x = "tl_sat_pct", y = "target", orient='h')

fig03 = set_image('Default Status Distribution According to Percent Satisfactory to Total Trade Lines', 
                  'Percent satisfactory to total trade tines', 'Default Status')

plt.savefig('../../images/boxplot1.png', dpi=150, format='png', bbox_inches='tight')

### Figure 04 Boxplot 2

fig04 = sns.boxplot(data=df, x = "tl_del60_cnt24", y = "target", orient='h')

fig04 = set_image('Default Status Distribution According to Number Trade Lines 60 Days 24 Months', 
                  'Number trade lines 60 days 24 months', 'Default Status')

plt.savefig('../../images/boxplot2.png', dpi=150, format='png', bbox_inches='tight')

### Figure 05 ROC Curve

# Creating empty dataframe
aux = pd.DataFrame()

# Formatting columns
aux['cumm_bad_perc']    = np.round(bp['cumm_bad_perc']*100, 2)
aux['cumm_good_perc']   = np.round(bp['cumm_good_perc']*100, 2)

fig05 = sns.lineplot(data=aux, x='cumm_bad_perc', y='cumm_good_perc', marker='o', markersize=10)

fig05 = set_image('ROC Curve', 'Cummulative Bad %', 'Cummulative Good %')

plt.savefig('../../images/roc_curve.png', dpi=150, format='png', bbox_inches='tight')
