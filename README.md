# Credit Scoring Project for a Bank

## Dashboard Page
https://credit-scoring-bank-caio-casagrande.streamlit.app/

## About Credit Scoring

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

## Business Problem

**Business Challenge: Developing a Risk Model**

The bank is embarking on a project to create an internal risk model that will guide lending decisions for subprime mortgages. The primary goal is to optimize profitability, but in addition to that the bank aims to strike a balance between profitability and market expansion, aligning with its strategic objectives of a business on the rise. This approach ensures that, while maximizing profits is a priority, the institution also seeks to grow its market presence in the domain of subprime mortgages. We will take into consideration the following key financial parameters:

- A profit of $100 is expected from each good customer;
- On the other hand, a loss of $500 is expected from each bad customer.

## Results

|decile|count_of_decile|sum_of_actual_outcome|min_prob_good|good|cumm_good|cumm_bad|cumm_good_perc|cumm_bad_perc | cumm_bad_avoided_perc| profit_to_business|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |									
| 1| 60| 0| 97.5%| 60| 60| 0| 12.4%| 0.0%| 100.0%| $6,000 |
| 2| 60| 0| 96.3%| 60| 120| 0| 24.9%| 0.0%| 100.0%| $12,000 |
| 3| 60| 4| 94.5%| 56| 176| 4| 36.5%| 3.4%| 96.6%| $15,600 |
| 4| 60| 8| 92.8%| 52| 228| 12| 47.3%| 10.2%| 89.8%| $16,800 |
| 5| 60| 9| 90.9%| 51| 279| 21| 57.9%| 17.8%| 82.2%| $17,400 |
| 6| 60| 7| 87.6%| 53| 332| 28| 68.9%| 23.7%| 76.3%| $19,200 |
| 7| 60| 12| 82.4%| 48| 380| 40| 78.8%| 33.9%| 66.1%| $18,000 |
| 8| 60| 13| 74.2%| 47| 427| 53| 88.6%| 44.9%| 55.1%| $16,200 |
| 9| 60| 34| 58.4%| 26| 453| 87| 94.0%| 73.7%| 26.3%| $1,800 |
| 10| 60| 31| 1.6%| 29| 482| 118| 100.0%| 100.0%| 0.0%| $-10,800 |

## Business Insights

**Elevating Profitability and Market Presence Through Informed Decision-Making**

This bank's Credit Score Project aimed to build an in-house risk model for subprime mortgages, with profitability as the primary focus. Each good customer is expected to bring in a profit of $100, while the cost of a bad customer is a significant $500. However, the bank's strategy extends beyond pure profit maximization, it encompasses a delicate balance between profitability and market expansion to align with the institution's overall business objectives.

The analysis of credit score deciles has illuminated key insights. The top-performing deciles 1, 2, and 3 are the ideal choice for conservative loan approval strategies. Selecting these top deciles enables businesses to avoid 96.6% of risky customers. Yet, the bank's strategic decision-making doesn't end at profitability. It extends to managing the level of exposure to risky customers, making it a calculated trade-off, recognizing that while potential profits may be somewhat sacrificed, market expansion and customer reach are the key to long-term success. The sixth decile emerges as the peak of profitability, but strategic exploration of adjacent deciles provides opportunities to maximize market share and broaden the customer base. 

This Credit Score Project equips the bank with the tools to make informed lending decisions. However, the path forward lies in the hands of the bank itself, trying to balance between maximizing profits and expanding its market presence.

## Streamlit Tabs
- About Credit Score - What is a Credit Score, Key Factors, Challenges, and Business Problem.
- Project - Models and results for the project.
- Business Performance - Results, numbers and profit margins for each decile calculated in the project.
- Conclusion - Business insights on the results of the Credit Score Project.
