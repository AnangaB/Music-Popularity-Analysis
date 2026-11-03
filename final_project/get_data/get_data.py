import pandas as pd

def get_data(filepath):
    data = pd.read_csv(filepath, index_col='track_id')

    features_numerical = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms", "lyric_words"]
    feature_categorical = ["playlist_genre", "key", "mode"]
    target = "track_popularity_categorical"

    data_filters = {
        "without_target": data[features_numerical].copy(),
        "without_target_with_categorical": data[features_numerical + feature_categorical].copy(),
        "with_target": data[features_numerical + [target]].copy(),
        "with_target_with_categorical": data[features_numerical + feature_categorical + [target]].copy(),
        "DEBUG_with_target_with_categorical_and_track_popularity": data[features_numerical + feature_categorical + [target] + ["track_popularity"]].copy(),
    }

    return data_filters

if __name__ == "__main__":
    data = get_data("./../../data/preprocessed//spotify_songs_preprocessed.csv")
    for key, val in data.items():
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(key)
        print(val)
        if key == "DEBUG_with_target_with_categorical_and_track_popularity":
            print(val['track_popularity'].value_counts())
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
