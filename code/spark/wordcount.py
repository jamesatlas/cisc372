from functools import reduce

#canonical WordCount example for map reduce using python's functional
# programming built-in library
                
def mapper(name):
  return (name, 1) # python tuple

def reducer(partial_counts, name_value):
  name = name_value[0]
  count = name_value[1]
  if name in partial_counts:
    partial_counts[name] = partial_counts[name] + count
  else:
    partial_counts[name] = count
  return partial_counts
  
with open("nonsense.txt","r") as f:
  # technically this is step 0, distributing the raw bytes/lines of the file
  #  is done as a parallel (distributed file read) in Spark
  lines = f.read().split()
  step1 = map(mapper, lines)
  step2 = reduce(reducer, step1, {})
  # in spark there will be a step3
  # step3 = collect partial results from data set
  print(step2)