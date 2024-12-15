import streamlit as st
import streamlit.components.v1 as components
from music_recommendations import recommendation_song

# Streamlit UI
st.set_page_config(layout="wide")

# Layout with two columns
col1, col2 = st.columns([2, 1])

# Right column: User input
with col2:
    st.title("Select Your Emotion")
    amount_of_song = st.select_slider("Amount", options=["3", "5", "7"])  # Allow user to select amount of songs
    user_emotion = st.selectbox("Choose an emotion:", ["Calm", "Joyful", "Angry", "Sad", "Energetic"])
    
    if st.button("Submit"):
        try:
            # Call the function to predict emotion and recommend songs
            recommendations = recommendation_song(user_emotion, int(amount_of_song))  
            
            # Check if recommendations is valid
            if recommendations is not None and not recommendations.empty:
                # Add a URI column with Spotify links if 'track_album_id' exists
                if "track_album_id" in recommendations.columns:
                    # clickable URI column with HTML
                    recommendations["URI"] = recommendations["track_album_id"].apply(
                        lambda x: f'<a href="https://open.spotify.com/album/{x}" target="_blank">Spotify Link</a>'
                    )
                else:
                    st.error("The column 'track_album_id' is missing in the recommendations.")

                # Save recommendations to session state
                st.session_state.recommendations = recommendations
                st.session_state.start_track_i = 0 
            else:
                st.error("No recommendations found for the selected emotion.")
        except ValueError as e:
            st.error(f"Error: {e}")

# Left column: Display recommendations
with col1:
    st.title("Recommended Songs")
    if "recommendations" in st.session_state:
        recommendations = st.session_state.recommendations
        if recommendations is not None and not recommendations.empty:
            # Add a URI column with Spotify links
            if "URI" not in recommendations.columns:
                recommendations["URI"] = recommendations["track_album_id"].apply(
                    lambda x: f'<a href="https://open.spotify.com/album/{x}" target="_blank">Spotify Link</a>'
                )

            # Add an ID column for the table
            recommendations = recommendations.reset_index(drop=True).reset_index().rename(columns={"index": "id"})

            # Display the table with clickable links
            st.write(f"If you're {user_emotion}, you can listen to these song:")
            html_table = recommendations[["id", "track_name", "track_artist", "URI"]].to_html(escape=False, index=False)
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.write("No recommendations found for the selected emotion.")
    else:
        st.write("Select an emotion and press Submit to see recommendations.")
