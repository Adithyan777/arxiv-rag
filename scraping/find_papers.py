import arxivscraper
import pandas as pd

cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')

# scraper = arxivscraper.Scraper(category='cs', date_from='2025-05-01',date_until='2025-05-10',filters={'categories':['cs.cv']})
# output = scraper.scrape()   

# df = pd.DataFrame(output,columns=cols)
# df.to_csv('arxiv_scrape_results.csv', index=False)
# print("Scraping completed. Results saved to 'arxiv_scrape_results.csv'.")

df = pd.read_csv('arxiv_scrape_results.csv', usecols=cols)

# make a new file where created date is between 2025-05-01 and 2025-05-10
filtered_df = df[df['created'].between('2025-05-01', '2025-05-10')]
filtered_df.to_csv('arxiv_scrape_results_filtered.csv', index=False) # 492 papers
print("Filtered results saved to 'arxiv_scrape_results_filtered.csv'.")