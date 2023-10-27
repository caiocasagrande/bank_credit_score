
##### Price Elasticity of Demand #####

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
from PIL                        import Image

import streamlit                as st
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

##### 2. Loading Data #####

### Loading Data

### Processed datasets 

df          = pd.read_csv('data/processed/processed_data.csv')
df_metrics  = pd.read_csv('data/processed/df_metrics.csv')
df_results  = pd.read_csv('data/processed/df_results.csv')
df_output   = pd.read_csv('data/processed/classifier_output.csv')

df_metrics.set_index('decile', inplace=True)
df_results.set_index('decile', inplace=True)

### Images

fig01 = Image.open('images/heatmap.png')
fig02 = Image.open('images/default_status.png')
fig03 = Image.open('images/boxplot1.png')
fig04 = Image.open('images/boxplot2.png')
fig05 = Image.open('images/roc_curve.png')

##### 3. Streamlit App #####

# Set the background color of the Streamlit app
st.set_page_config(layout='wide', page_title='Bank Credit Score', page_icon='📊', initial_sidebar_state='expanded')

# st.sidebar.markdown('# Credit Score')
st.sidebar.markdown("""---""")
st.sidebar.markdown(
    """
    **Business Challenge: Developing a Credit Score Model**
    
    In this Data Science project, the focus is on the development of an internal risk model by a banking institution to refine lending decisions. 

    Concentrating on profitability as the primary goal, the project also underscores the bank's commitment to achieving a delicate equilibrium between maximizing profits and expanding its market-share presence. 

    This project demonstrates the crucial role of Data Science and analytics in driving strategic decisions within a financial institution, shedding light on the potential advantages of such analysis for businesses aiming to thrive and grow in a dynamic market environment.
    """
)
st.sidebar.markdown("""---""")
st.sidebar.markdown('Powered by [Caio Casagrande](https://www.linkedin.com/in/caiopc/)')
st.sidebar.markdown('Github [Notebook](https://github.com/caiocasagrande/bank_credit_score/blob/main/notebooks/bank_credit_scoring.ipynb)')

st.header('Credit Score Project')

tab1, tab2, tab3, tab4 = st.tabs(['About Credit Score', 
                                  'Project', 
                                  'Business Performance', 
                                  'Conclusion'])

with tab1:
    st.subheader('Business Description')
    st.write("""
        **Understanding Credit Scores**

        A credit score is a statistical analysis conducted by banks and financial institutions to assess the creditworthiness of borrowers. It plays a crucial role in the lending decision process, helping financial entities to measure the risk associated with extending loans.

        **Key Factors Impacting Credit Scores**

        Several factors influence the computation of a credit score, which serves as a numerical representation of an individual's credit history. These factors include the borrower's repayment history, the duration of their credit history, the number of previous credit inquiries, and the amount of active credit cards and loans. When combined, these elements yield a credit score, a tool that banks employ to make informed lending decisions.

        **Challenges in Credit Scoring**

        Despite its utility, credit scoring presents several challenges that banks must be aware of:

        1. *Limited Credit History*: Not all borrowers possess an extensive credit history, making it challenging to establish a robust credit score.

        2. *Bank Size and Strategy*: The size and strategic orientation of a bank also influence credit scoring. For instance, a borrower with a strong credit score may prefer larger lenders, complicating the decision-making process for smaller institutions.

        3. *Bank Objectives*: Banks define their objectives based on factors like risk minimization, profit maximization, or market expansion. Public banks may prioritize lower-risk applicants, offering fewer incentives to those with lower credit scores. In contrast, private banks seek to optimize their credit score tolerances. New financial institutions, aiming to expand their market share, are often open to applicants with low or nonexistent credit scores.

        4. *Customized Approaches*: Each business adjust its approach to assessing loan applications and making lending decisions in alignment with its unique business strategy.
        """)
    
    st.subheader('Business Problem')
    st.write("""
        **Business Challenge: Developing a Risk Model**

        The bank is embarking on a project to create an internal risk model that will guide lending decisions for subprime mortgages. The primary goal is to optimize profitability, but in addition to that the bank aims to strike a balance between profitability and market expansion, aligning with its strategic objectives of a business on the rise. This approach ensures that, while maximizing profits is a priority, the institution also seeks to grow its market presence in the domain of subprime mortgages. We will take into consideration the following key financial parameters:

        - A profit of $100 is expected from each good customer;
        - On the other hand, a loss of $500 is expected from each bad customer.
        """
        )

with tab2:
    with st.container():
        st.markdown('### Dataset Overview')

        st.dataframe(df, use_container_width=True)

    with st.container():
        st.markdown("""---""")
        st.markdown('### Visual Analysis')

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(fig01, use_column_width=True)

        with col2:
            st.image(fig02, use_column_width=True)
            st.image(fig03, use_column_width=True)
            st.image(fig04, use_column_width=True)

    with st.container():
        st.markdown("""---""")
        st.markdown('### Logistic Regression Outputs')
        st.dataframe(df_output, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                        **Classification Report**

                |        | Precision | Recall | F1-Score | Support |
                |--------|-----------|--------|----------|---------|
                |   0    |   0.83    |  0.97  |   0.89   |   482   |
                |   1    |   0.57    |  0.18  |   0.27   |   118   |
                |        |           |        |          |         |
                |Accuracy|           |        |   0.81   |   600   |
                |Macro Avg|   0.70   |  0.57  |   0.58   |   600   |
                |Weighted Avg| 0.78  |  0.81  |   0.77   |   600   |

                """)

        with col2:
            st.markdown("""
                        **Confusion Matrix**

            |           | Predicted 0 | Predicted 1 |
            |-----------|-------------|-------------|
            | Actual 0  |     466     |     16      |
            | Actual 1  |     97      |     21      |

            """)

    with st.container():
        st.markdown("""---""")
        st.markdown('### Business Metrics')

        st.dataframe(df_metrics, use_container_width=True)

with tab3:
    with st.container():
        st.markdown('### Business Results')

        st.dataframe(df_results, use_container_width=True)
        st.markdown("""---""")

        st.markdown('### Business Discussion')
        st.write("""
            In deciles 1, 2, and 3, we find our top-performing customers. For businesses seeking a highly conservative loan approval strategy, these deciles offer an ideal choice. These top deciles exhibit an impressive minimum non-default probability of 94.5%, with only 4 out of 180 borrowers expected to default. By selecting these top 3 deciles, a business can effectively avoid 96.6% of high-risk customers.

            However, it's essential for a bank to align its strategy with the business's objectives. In other words, the bank must determine the desired level of exposure to risky customers. As we move down the decile sets, businesses can access a larger pool of good customers, but this expansion comes at the trade-off of increasing exposure to higher-risk borrowers. 
            
            Therefore, the selection of deciles is a strategic decision that must be finely tuned to strike the right balance between growing the customer base and managing risk effectively.
                 
            As evident from the data, the peak of profitability is achieved in the sixth decile, delivering a profit of $19,200. Nevertheless, banks may opt to explore opportunities in the adjacent deciles, even while some potential profits may be sacrificed. Take, for example, the seventh decile, characterized by a cutoff probability of 82.4%. 
            
            This strategic decision allows banks to not only maximize their market share but also broadens their customer base,  with a consideration of the trade-off in potential profitability.
            """)
        st.markdown("""---""")
        st.image(fig05, use_column_width=True)

with tab4:
    st.markdown('### Conclusion')
    st.write("""
        **Elevating Profitability and Market Presence Through Informed Decision-Making**

        This bank's Credit Score Project aimed to build an in-house risk model for subprime mortgages, with profitability as the primary focus. Each good customer is expected to bring in a profit of 100 dollars, while the cost of a bad customer is a significant 500 dollars. However, the bank's strategy extends beyond pure profit maximization, it encompasses a delicate balance between profitability and market expansion to align with the institution's overall business objectives.

        The analysis of credit score deciles has illuminated key insights. The top-performing deciles 1, 2, and 3 are the ideal choice for conservative loan approval strategies. Selecting these top deciles enables businesses to avoid 96.6% of risky customers. Yet, the bank's strategic decision-making doesn't end at profitability. 
        
        It extends to managing the level of exposure to risky customers, making it a calculated trade-off, recognizing that while potential profits may be somewhat sacrificed, market expansion and customer reach are the key to long-term success. The sixth decile emerges as the peak of profitability, but strategic exploration of adjacent deciles provides opportunities to maximize market share and broaden the customer base. 

        This Credit Score Project equips the bank with the tools to make informed lending decisions. However, the path forward lies in the hands of the bank itself, trying to balance between maximizing profits and expanding its market presence.
        """)
    st.markdown("""---""")
    st.markdown('### Final Results')
    st.dataframe(df_results, use_container_width=True)