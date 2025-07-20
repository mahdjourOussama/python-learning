# list variable
# a list is a collection of elements that can be of different types
my_list = [1, 2, 3, 4, 5, 5, 5, 5, 5, "ousssama", True, 1.86]
# print("list:", my_list)

# how to handle lists

# get the element from list
print("first element of list:", my_list[0])
# get the last element from list
print("last element of list:", my_list[-1])
# get the first 5 elements from list
print("first 5 elements of list:", my_list[:5])
# get a range of element from list
print("elements from index 2 to 5:", my_list[2:6])

# add element to list
my_list.append("new element")
print("list after adding new element:", my_list)
# remove element from list
my_list.remove("ousssama")
print("list after removing ousssama:", my_list)
# remove element from list by index
my_list.pop(0)
print("list after removing first element:", my_list)

# how to change elements in list using index
my_list[0] = "changed element"
print("list after changing first element:", my_list)

#  accesing the length of the list
print("length of list:", len(my_list))
