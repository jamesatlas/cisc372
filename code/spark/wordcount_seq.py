#canonical WordCount example using sequential iteration and mutation
  
with open("nonsense.txt","r") as f:
  lines = f.read().split()
  partial_counts = {}
  for name in lines:
    if name in partial_counts:
      partial_counts[name] = partial_counts[name] + 1
    else:
      partial_counts[name] = 1
  print(partial_counts)