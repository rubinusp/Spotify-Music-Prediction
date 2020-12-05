import dataextractor as de

extr = de.DataExtractor()
arr = extr.load_json().to_array()

print(len(arr))
print(len(arr[0]))

# Acousticiness [0]
# Dancibility [1]
# Duration [2]
# Energy [3]
# Explicit [4]
# Instrumentalness [5]
# Key [6]
# Liveness [7]
# Loudness [8]
# Mode [9]
# Popularity [10]
# Speechiness [11]
# Tempo [12]
# Time signature [13]
# Valence [14]
