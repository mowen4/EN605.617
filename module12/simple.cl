//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(const uint n, __global float* buffer, __local float* partialSum, __local float* localSum)
{
	uint global_id = get_global_id(0);
	uint global_size = get_global_size(0);
	uint local_id = get_local_id(0);
	uint group_size = get_local_size(0);
	float temp;

	localSum[local_id] = buffer[global_id] + buffer[global_id+global_size];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(uint stride=group_size/2; stride > 1; stride >>=1) {
	if (local_id <stride)
		localSum[local_id] += localSum[local_id + stride];
		temp = localSum[local_id] / 2;
		localSum[local_id] = temp;
	barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (local_id == 0)
		buffer[get_group_id(0)] = localSum[0] + localSum[1];
		
}
