import numpy

def iterate(Z):
    # find number of neighbours that each square has
    N = numpy.zeros(Z.shape)
    N[1:, 1:] += Z[:-1, :-1]
    N[1:, :-1] += Z[:-1, 1:]
    N[:-1, 1:] += Z[1:, :-1]
    N[:-1, :-1] += Z[1:, 1:]
    N[:-1, :] += Z[1:, :]
    N[1:, :] += Z[:-1, :]
    N[:, :-1] += Z[:, 1:]
    N[:, 1:] += Z[:, :-1]
    # a live cell is killed if it has fewer 
    # than 2 or more than 3 neighbours.
    part1 = ((Z == 1) & (N < 4) & (N > 1)) 
    # a new cell forms if a square has exactly three members
    part2 = ((Z == 0) & (N == 3))
    return (part1 | part2).astype(int)

Z = numpy.array([[0,0,0,0,0,0],
                 [0,0,0,1,0,0],
                 [0,1,0,1,0,0],
                 [0,0,1,1,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0]])
print 'Initial state:'
print Z[1:-1,1:-1]
for i in range(4):
    Z = iterate(Z)
print 'Final state:'
print Z[1:-1,1:-1]
