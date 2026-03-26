
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai import Agent, RunContext, WebSearchTool
from dotenv import load_dotenv
import edgar
import os
import time

from src.models.sec_balance_sheet import SecBalanceSheet

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# set user identity for edgar
edgar.set_identity("user@example.com")

# MODEL_NAME = 'openai/gpt-oss-120b:free'
MODEL_NAME = 'meta-llama/llama-3.1-8b-instruct'
model = OpenRouterModel(
    MODEL_NAME,
    provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
)

SYSTEM_PROMPT = ('You are a financial analyst assistant that provides information about a company\'s financial health based on its SEC filings. '
                 'You can retrieve the company\'s balance sheet and other relevant financial data to help users make informed investment decisions. '
                 'When a user asks for information about a company, you will use the provided ticker symbol to fetch the latest balance sheet and other financial data from the SEC filings. '
                 'You will then structure the information in a clear and concise manner, highlighting key financial metrics quantities. ')
                #  'You can use the WebSearchTool to gather additional information about the company if needed, but your primary source of financial data should be the SEC filings.')

sec_agent = Agent(  
    model,
    deps_type=str,
    output_type=SecBalanceSheet,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)


@sec_agent.tool
async def sec_info(ctx: RunContext[str], ticker: str) -> dict:  
    """get SEC information for a given ticker"""
    company = edgar.Company(ticker)
    statement = company.get_financials().balance_sheet()
    return SecBalanceSheet.from_edgar(
        ticker=ticker,
        company_name=company.name,
        statement=statement,
    ).model_dump(mode="json")

# Run the agent
start_time = time.perf_counter()
ticker = "AAPL"
result = sec_agent.run_sync(
    f'What is the current financial health of {ticker} based on its latest SEC filings?',
    deps=ticker,
    # builtin_tools=[WebSearchTool()]
)
end_time = time.perf_counter()

print(result.output)  
print(f"Execution time: {end_time - start_time:.2f} seconds")
