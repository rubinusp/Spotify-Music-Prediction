import dataextractor as de

extr = de.DataExtractor()
arr = extr.load_json().to_array()

print(len(arr))
print(len(arr[0]))