model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{4, 3, 2}")
axis = Parameter("axis", "TENSOR_INT32", "{2}", [0, 2])
keepDims = True
output = Output("output", "TENSOR_FLOAT32", "{1, 3, 1}")

model = model.Operation("REDUCE_MAX", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
           13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]}

output0 = {output: # output 0
          [20, 22, 24]}

# Instantiate an example
Example((input0, output0))
