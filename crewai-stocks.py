import os
from datetime import datetime

from dotenv import load_dotenv
import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # To deploy in streamlit, use this line instead load dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetch stock prices to {ticket} from the last year about a specific stock of the Yahoo Finance API.",
    func=lambda ticket: fetch_stock_price(ticket),
)


stock_price_analyst = Agent(
    role="Senior Stock Price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're a highly experienced in analyzing the price of an specific stock and make predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False,
)

get_stock_price = Task(
    description="Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output=""" Specify the current trend stock price - up, down or sideways.
    eg. stock= 'APPL, price UP' """,
    agent=stock_price_analyst,
)

search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

news_analyst = Agent(
    role="Stock News Analyst",
    goal=""" Create a short summary of the market related to the stock {ticket} company. 
    Specify the current trend - up, down or sideways with the news context. 
    For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed. """,
    backstory=""" You-re highly experienced in analyzing the market and news and have tracked assest for more than 10 years.
    You're also master level analysts in the tradicional markets and have deep understanding of human psychology.
    You understand news, theirs titles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles. """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[search_tool],
    allow_delegation=False,
)

get_news = Task(
    description=f"""Take the stock and always include BTC to it (if no request).
    Use the search tool to search each one individually.
    The request date is {datetime.now()}.
    Compose the results into a helpfull report. """,
    expected_output=""" A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. 
    Use format:
        <STOCK ASSET>
        <SUMARRY BASED ON NEWS>
        <TREND PREDICTION>
        <FEAR/GREED SCORE>
    """,
    agent=news_analyst,
)

stock_analyst_write = Agent(
    role="Senior Writer",
    goal="""Analyze the trends price and news and write and insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.""",
    backstory=""" You're widely accpeted as athe best stock analyst in the market. 
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories - eg. cycle theory fundamental analyses.
    You're able to hold multiple opnions when analyzing anything.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)

write_analysis = Task(
    description=""" Use the stock price and the stock news report to create an analysis and write the newsletter 
    about the {ticket} company that is brief and highligts the most important points.
    Focus on the sock price trend, news and fear/greed score. What are the near future considerations? 
    Include the previous analysis of stock trend and news summary. """,
    expected_output=""" An eloquent 3 paragraph newsletter formated as markdown in an easy readable manner. It should contain:
        - 3 bullets executive summary.
        - Introduction - set the overerall picture and spike up the interest.
        - main part provides the meat of the analysis including the news summary and feed/greed scores.
        - summary - key facts and concrete future trend prediction - up, down or sideways.
    """,
    agent=stock_analyst_write,
    context=[get_stock_price, get_news],
)

crew = Crew(
    agents=[stock_price_analyst, news_analyst, stock_analyst_write],
    tasks=[get_stock_price, get_news, write_analysis],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=True,
    manager_llm=llm,
    max_iter=15,
)

with st.sidebar:
    st.header("Enter the ticket stock")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={"ticket": topic})
        st.write(results["final_output"])
