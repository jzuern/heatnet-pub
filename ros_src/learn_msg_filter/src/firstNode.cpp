#include "ros/ros.h"
#include <sstream>
#include <bits/stdc++.h>
#include "learn_msg_filter/NewString.h"


int main(int argc, char** argv) {

	ros::init(argc, argv, "firstNode");
	ros::NodeHandle nh;
	ROS_INFO_STREAM("First node started.");
	ros::Publisher pub = nh.advertise<learn_msg_filter::NewString>("rgb_0", 5);
	ros::Rate loop_rate(30);

	while(ros::ok()) {

		learn_msg_filter::NewString msg;
		
		msg.header.stamp = ros::Time::now();
		msg.header.frame_id = "/myworld";


		std::stringstream ss;
		ss << "RGB image 30 hz\n";
		msg.st = ss.str();
		
		pub.publish(msg);

		ros::spinOnce();

		loop_rate.sleep();

	}

	return 0;
	
}

