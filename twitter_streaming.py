#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
consumer_key = 'v1qY87LA5VzjRC85NSjoIgvtJ'
consumer_secret = '9dh1s11jA2szncqsD0mGE2B0mbPaR66FDBb5eP60LeOrY7u14e'
# This access token can be used to make API requests on your own account's behalf.
access_token = '2410927159-YywJdK9YyaZIPFWzELulc123R0HpCAZ3jZ7E256'
access_token_secret = 'ZVt6uFcoyvcQhfciKJT8rbJaNKnzenIm8kh5k4xrJNsIk'


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return(True)

    def on_error(self, status):
        print(status)

if __name__ == '__main__':

    #This handles Twitter authentification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=['@SouthwestAir', '#SouthwestAir', "#Delta", "@Delta", "@United", "#United", "@DeltaAssist", "#DeltaAssist", "@AmericanAir", "#AmericanAir", "@JetBlue", "#JetBlue", "@Airbnb", "#Airbnb", "@AirbnbHelp", "#AirbnbHelp", "@COMCAST", "#COMCAST", "@comcastcares", "#comcastcares"])