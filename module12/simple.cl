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

__kernel void square(__global * buffer)
{
	size_t id = get_global_id(0);
	int sum = 0;
	
	sum += buffer[id] / 16;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	buffer[0] = sum;
}
