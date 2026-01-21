from pydantic import BaseModel, Field

class RequestModel(BaseModel):
    query: str

class Screener_Query(BaseModel):
    screener_query: str = Field(description="Query for screener.in, using about 5-10 (or more) queries given to you below")

keywords = """
Search Query
You can customize the query using the variables/parameters given below from screener.in:

.....................
Sales
OPM
Profit after tax
Market Capitalization
Sales latest quarter
Profit after tax latest quarter
YOY Quarterly sales growth
YOY Quarterly profit growth
Price to Earning
Dividend yield
Price to book value
Return on capital employed
Return on assets
Debt to equity
Return on equity
EPS
Debt
Promoter holding
Change in promoter holding
Earnings yield
Pledged percentage
Industry PE
Sales growth
Profit growth
Current price
Price to Sales
Price to Free Cash Flow
EVEBITDA
Enterprise Value
Current ratio
Interest Coverage Ratio
PEG Ratio
Return over 3months
Return over 6months
..............................
Return on capital employed
EPS
Change in promoter holding
Sales last year
Operating profit last year
Other income last year
EBIDT last year
Depreciation last year
EBIT last year
Interest last year
Profit before tax last year
Tax last year
Profit after tax last year
Extraordinary items last year
Net Profit last year
Dividend last year
Material cost last year
Employee cost last year
OPM last year
NPM last year
Operating profit
Interest
Depreciation
EPS last year
EBIT
Net profit
Current Tax
Tax
Other income
Last annual result date
..........................
Sales preceding year
Operating profit preceding year
Other income preceding year
EBIDT preceding year
Depreciation preceding year
EBIT preceding year
Interest preceding year
Profit before tax preceding year
Tax preceding year
Profit after tax preceding year
Extraordinary items preceding year
Net Profit preceding year
Dividend preceding year
OPM preceding year
NPM preceding year
EPS preceding year
Sales preceding 12months
Net profit preceding 12months
.............................
Sales growth 3Years
Sales growth 5Years
Profit growth 3Years
Profit growth 5Years
Sales growth 10years median
Sales growth 5years median
Sales growth 7Years
Sales growth 10Years
EBIDT growth 3Years
EBIDT growth 5Years
EBIDT growth 7Years
EBIDT growth 10Years
EPS growth 3Years
EPS growth 5Years
EPS growth 7Years
EPS growth 10Years
Profit growth 7Years
Profit growth 10Years
Change in promoter holding 3Years
Average Earnings 5Year
Average Earnings 10Year
Average EBIT 5Year
Average EBIT 10YearDebt
Equity capital
Preference capital
Reserves
Secured loan
Unsecured loan
Balance sheet total
Gross block
Revaluation reserve
Accumulated depreciation
Net block
Capital work in progress
Investments
Current assets
Current liabilities
Book value of unquoted investments
Market value of quoted investments
Contingent liabilities
Total Assets
Working capital
Lease liabilities
Inventory
Trade receivables
Face value
Cash Equivalents
Advance from Customers
Trade Payables
.........................
Number of equity shares preceding year
Debt preceding year
Working capital preceding year
Net block preceding year
Gross block preceding year
Capital work in progress preceding year
..............................
Working capital 3Years back
Working capital 5Years back
Working capital 7Years back
Working capital 10Years back
Debt 3Years back
Debt 5Years back
Debt 7Years back
Debt 10Years back
Net block 3Years back
Net block 5Years back
Net block 7Years back
................................
Cash from operations last year
Free cash flow last year
Cash from investing last year
Cash from financing last year
Net cash flow last year
Cash beginning of last year
Cash end of last year
.........................
Free cash flow preceding year
Cash from operations preceding year
Cash from investing preceding year
Cash from financing preceding year
Net cash flow preceding year
Cash beginning of preceding year
Cash end of preceding year
.............................
Free cash flow 3years
Free cash flow 5years
Free cash flow 7years
Free cash flow 10years
Operating cash flow 3years
Operating cash flow 5years
Operating cash flow 7years
Operating cash flow 10years
Investing cash flow 10years
Investing cash flow 7years
Investing cash flow 5years
Investing cash flow 3years
Cash 3Years back
Cash 5Years back
Cash 7Years back
.................................
Price to Earning
Dividend yield
Price to book value
Return on assets
Debt to equity
Return on equity
Promoter holding
Earnings yield
Pledged percentage
Industry PE
Enterprise Value
Number of equity shares
Price to Quarterly Earning
Book value
Inventory turnover ratio
Quick ratio
Exports percentage
Piotroski score
G Factor
Asset Turnover Ratio
Financial leverage
Number of Shareholders
Unpledged promoter holding
Return on invested capital
Debtor days
Industry PBV
Credit rating
Working Capital Days
Earning Power
Graham Number
Cash Conversion Cycle
Days Payable Outstanding
Days Receivable Outstanding
Days Inventory Outstanding
Public holding
FII holding
Change in FII holding
DII holding
Change in DII holding
......................
Book value preceding year
Return on capital employed preceding year
Return on assets preceding year
Return on equity preceding year
Number of Shareholders preceding quarter
.........................
Average return on equity 5Years
Average return on equity 3Years
Number of equity shares 10years back
Book value 3years back
Book value 5years back
Book value 10years back
Inventory turnover ratio 3Years back
Inventory turnover ratio 5Years back
Inventory turnover ratio 7Years back
Inventory turnover ratio 10Years back
Exports percentage 3Years back
Exports percentage 5Years back
Average 5years dividend
Average return on capital employed 3Years
Average return on capital employed 5Years
Average return on capital employed 7Years
Average return on capital employed 10Years
Average return on equity 10Years
Average return on equity 7Years
Return on equity 5years growth
OPM 5Year
OPM 10Year
Number of Shareholders 1year back
Average dividend payout 3years
Average debtor days 3years
Debtor days 3years back
Debtor days 5years back
Return on assets 5years
Return on assets 3years
Historical PE 3Years
Historical PE 10Years
Historical PE 7Years
Historical PE 5Years
Market Capitalization 3years back
Market Capitalization 5years back
Market Capitalization 7years back
Market Capitalization 10years back
Average Working Capital Days 3years
Change in FII holding 3Years
Change in DII holding 3Years
................................
Operations:
+
-
/
*
>
<
AND
................................

Note:
1. You must always only write queries that will screen out small/bad companies and only give fundamnetally strong companies.
2. Stick to writing diverse queries that reflect your deep understanding of queries of screener.in
"""
