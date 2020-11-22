#!/usr/bin/env python
# coding: utf-8

# <h1>1. Introduction:<h1/>

# The battle of neighborhoods has been implemented for the exploration of data using Segmenting and Clustering techniques applied to the neighborhoods in Toronto. The objective of Applied Data ScienceCapstone is given as follows:
# 
# 1.To learn about clustering and k-means clustering in particular.<br>
# 2.To showcase this project in the form of the public repository using the GitHub platform.<br>
# 3.To learn how to use the Foursquare API and clustering to segment and cluster the neighborhoods in New York City.<br>
# 4.To learn how to use the Beautifulsoup Python package to scrape websites and parse HTML code.<br>
# 5.To apply the skills acquired so far in this course to segment and cluster neighborhoods in the city of Toronto ,subject of the following notebook.

# <h1>2. Scrap content from wiki page<h1/>

# <h5>Import necessary packages.<h5/>

# In[60]:


import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe
get_ipython().system('pip install folium==0.5.0')
import folium # map rendering library

# import k-means from clustering stage
from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# <h5>Scrape the "raw" table.<h5/>

# In[61]:


source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
soup = BeautifulSoup(source, 'lxml')

table = soup.find("table")
table_rows = table.tbody.find_all("tr")

res = []
for tr in table_rows:
    td = tr.find_all("td")
    row = [tr.text for tr in td]
    
    # Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
    if row != [] and row[1] != "Not assigned":
        # If a cell has a borough but a "Not assigned" neighborhood, then the neighborhood will be the same as the borough.
        if "Not assigned" in row[2]: 
            row[2] = row[1]
        res.append(row)

# Dataframe with 3 columns
df = pd.DataFrame(res, columns = ["PostalCode", "Borough", "Neighborhood"])
df.head()


# <h5>Remove "\n" at the end of each string in the three columns<h5/>

# In[62]:


df["Neighborhood"] = df["Neighborhood"].str.replace("\n","")
df["PostalCode"] = df["PostalCode"].str.replace("\n","")
df["Borough"] = df["Borough"].str.replace("\n","")
df.head()


# <h5>Identify and Drop missing Values <h5/>

# In[63]:


df.replace("Not assigned",np.nan,inplace=True)
df.head(5)


# In[64]:


# Let's simply drop whole row with Nan in "Neighborhood" column
df.dropna(subset=["Neighborhood"],axis=0,inplace=True)
#rest index
df.reset_index(drop=True,inplace=True)


# In[65]:


df.head(12)


# In[66]:


print("Shape: ", df.shape)


# <h1>3. Get the latitude and the longitude coordinates of each neighborhood.<h1/>

# <h5>We are not able to get the geohraphical coordinates of the neighborhoods using the Geocoder package, we use the given csv file instead.<h5/>

# In[67]:



# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IAM_SERVICE_ID': 'iam-ServiceId-f968dcdc-ea31-4313-9887-babf91d91441',
    'IBM_API_KEY_ID': '9rYvtpBrAnbT5g6pnfLXakSElQXMb04f6eKC9b-AfRhA',
    'ENDPOINT': 'https://s3.eu-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.cloud.ibm.com/oidc/token',
    'BUCKET': 'segmentingandclusteringneighborho-donotdelete-pr-ijbhtwqvmf3xbz',
    'FILE': 'Geospatial_Coordinates.csv'
}
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_0ca568d8dcb0447ab571f8ce182d09d0 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='9rYvtpBrAnbT5g6pnfLXakSElQXMb04f6eKC9b-AfRhA',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_0ca568d8dcb0447ab571f8ce182d09d0.get_object(Bucket='segmentingandclusteringneighborho-donotdelete-pr-ijbhtwqvmf3xbz',Key='Geospatial_Coordinates.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()


# <h5>Now the next step is to couple 2 dataframes "df" and "df_data_1" into one dataframe.<h5/>

# In[68]:


df_toronto = pd.merge(df, df_data_1, how='left', left_on = 'PostalCode', right_on = 'Postal Code')
# remove the "Postal Code" column
df_toronto.drop("Postal Code", axis=1, inplace=True)
df_toronto.head(12)


# <h3>4. Explore and cluster the neighborhoods in Toronto<h3/>

# <h5>Let's get the latitude and longitude values of Toronto.<h5/>

# In[69]:


address = "Toronto, ON"

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))


# <h5>Now let's Create a map of the whole Toronto City with neighborhoods superimposed on top.<h5/>
# 

# In[70]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
map_toronto


# <h5>Add markers to the map.<h5/>

# In[72]:


for lat, lng, borough, neighborhood in zip(
        df_toronto['Latitude'], 
        df_toronto['Longitude'], 
        df_toronto['Borough'], 
        df_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  

map_toronto


# <h5> We are going to work with only the boroughs that contain the word "Toronto".<h5/>

# In[73]:


# "denc" = [D]owntown Toronto, [E]ast Toronto, [N]orth Toronto, [C]entral Toronto
df_toronto_denc = df_toronto[df_toronto['Borough'].str.contains("Toronto")].reset_index(drop=True)
df_toronto_denc.head()


# <h5> Then let's Plot again the map and the markers for this region.<h5/>

# In[74]:


map_toronto_denc = folium.Map(location=[latitude, longitude], zoom_start=12)
for lat, lng, borough, neighborhood in zip(
        df_toronto_denc['Latitude'], 
        df_toronto_denc['Longitude'], 
        df_toronto_denc['Borough'], 
        df_toronto_denc['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto_denc)  

map_toronto_denc


# <h5>Define Foursquare Credentials and Version<h5/>

# In[75]:


CLIENT_ID = 'IOPMVZIUATAYXB0XJLL3BBG5ZCZABZG00QG3YEU25QTMQ1RA'
CLIENT_SECRET = 'KPZ4XE1JXUY4EDT2GIH5KIO5FRETATLJR15PB0LE5BJ5L5FB'
VERSION = '20180604'


# <h5>Explore the first neighborhood in our data frame "df_toronto"<h5/>

# In[76]:


neighborhood_name = df_toronto_denc.loc[0, 'Neighborhood']
print(f"The first neighborhood's name is '{neighborhood_name}'.")


# <h5>Get the neighborhood's latitude and longitude values.<h5/>

# In[77]:


neighborhood_latitude = df_toronto_denc.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_toronto_denc.loc[0, 'Longitude'] # neighborhood longitude value

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# <h5>Now, let's get the top 100 venues that are in The Harbourfront within a radius of 500 meters.<h5/>

# In[78]:


LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

# get the result to a json file
results = requests.get(url).json()


# <h5>Function that extracts the category of the venue<h5/>

# In[79]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# <h5>Now we are ready to clean the json and structure it into a pandas dataframe.<h5/>

# In[80]:


venues = results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# <h5>Explore neighborhoods in a part of Toronto City<h5>
#     <h5>We are working on the data frame df_toronto_denc. Recall that, this region contain DENC of Toronto where,
# 
# "DENC" = [D]owntown Toronto, [E]ast Toronto, [N]orth Toronto, [C]entral Toronto
# 
# First, let's create a function to repeat the same process to all the neighborhoods in DENC of Toronto.<h5/>

# In[81]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# <h5>Now write the code to run the above function on each neighborhood and create a new dataframe called toronto_denc_venues<h5/>

# In[82]:


toronto_denc_venues = getNearbyVenues(names=df_toronto_denc['Neighborhood'],
                                   latitudes=df_toronto_denc['Latitude'],
                                   longitudes=df_toronto_denc['Longitude']
                                  )


# In[86]:


toronto_denc_venues.head()


# <h5>Let's check how many venues were returned for each neighborhood.<h5/>

# In[87]:


toronto_denc_venues.groupby('Neighborhood').count()


# <h5>Let's find out how many unique categories can be curated from all the returned venues<h5/>

# In[88]:


print('There are {} uniques categories.'.format(len(toronto_denc_venues['Venue Category'].unique())))


# <h5>Analyze Each Neighborhood<h5/>

# In[89]:


# one hot encoding
toronto_denc_onehot = pd.get_dummies(toronto_denc_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_denc_onehot['Neighborhood'] = toronto_denc_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_denc_onehot.columns[-1]] + list(toronto_denc_onehot.columns[:-1])
toronto_denc_onehot = toronto_denc_onehot[fixed_columns]

toronto_denc_onehot.head()


# <h5>Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category<h5/>

# In[90]:


toronto_denc_grouped = toronto_denc_onehot.groupby('Neighborhood').mean().reset_index()
toronto_denc_grouped.head()


# <h5>Check the 10 most common venues in each neighborhood.<h5/>

# In[91]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_denc_grouped['Neighborhood']

for ind in np.arange(toronto_denc_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_denc_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# <h5>Cluster neighborhoods<h5/>
#     
#    <h5>Let's run k-means to cluster the neighborhood into 5 clusters.<h5/>

# In[92]:


# set number of clusters
kclusters = 5

toronto_denc_grouped_clustering = toronto_denc_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_denc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# <h5>Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood<h5/>

# In[93]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_denc_merged = df_toronto_denc

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_denc_merged = toronto_denc_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_denc_merged.head() # check the last columns!


# <h5>Finally, let's visualize the resulting clusters<h5/>

# In[94]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        toronto_denc_merged['Latitude'], 
        toronto_denc_merged['Longitude'], 
        toronto_denc_merged['Neighborhood'], 
        toronto_denc_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


#  <h5>Examine Clusters<br>
# Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster.<h5/>

# <h5>Cluster 1<h5/>

# In[95]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 0, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# <h5>Cluster 2<h5/>

# In[97]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 1, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# <h5>Cluster 3<h5/>

# In[98]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 2, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# <h5>Cluster 4<h5/>

# In[99]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 3, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# <h5>Cluster 5<h5/>

# In[100]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 4, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# In[ ]:




