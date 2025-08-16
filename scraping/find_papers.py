import arxivscraper
import pandas as pd

START_DATE = '2025-05-01'
END_DATE = '2025-05-10'

cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')

scraper = arxivscraper.Scraper(category='cs', date_from=START_DATE, date_until=END_DATE, filters={'categories':['cs.cv']})
output = scraper.scrape()   

df = pd.DataFrame(output,columns=cols)

# Uncomment if you do not want to filter
# df.to_csv('arxiv_scrape_results.csv', index=False)
# print("Scraping completed. Results saved to 'arxiv_scrape_results.csv'.")

# make a new file where created date is between 2025-05-01 and 2025-05-10
filtered_df = df[df['created'].between(START_DATE, END_DATE)]
filtered_df.to_csv('arxiv_scrape_results_filtered.csv', index=False) # 492 papers
print("Filtered results saved to 'arxiv_scrape_results_filtered.csv'.")