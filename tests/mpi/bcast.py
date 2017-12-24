from mpi4py import MPI                                                     
                                                                           
comm = MPI.COMM_WORLD                                                      
rank = comm.Get_rank()                                                     
size = comm.Get_size()                                                     
                                                                           
if rank == 0:                                                              
    data = range(10)                                                       
    print("process {} bcast data {} to other processes".format(rank, data))
else:                                                                      
    data = None                                                            
data = comm.bcast(data, root=0)                                            
print("process {} recv data {}...".format(rank, data))