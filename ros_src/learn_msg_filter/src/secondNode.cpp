#include "ros/ros.h"
#include <bits/stdc++.h>
#include <sstream>
#include "learn_msg_filter/NewString.h"


int main(int argc, char** argv) {

	ros::init(argc, argv, "secondNode");
	ros::NodeHandle nh;
	ROS_INFO_STREAM("Second Node started");

	ros::Publisher pub = nh.advertise<learn_msg_filter::NewString>("ir_0", 5);
	ros::Rate loop_rate(60);

	while(ros::ok()) {
		learn_msg_filter::NewString msg;

		msg.header.stamp = ros::Time::now();
		msg.header.frame_id = "/robot";

		std::stringstream ss;
		ss << "IR image 60 hz!\n";
		msg.st = ss.str();
		pub.publish(msg);

		ros::spinOnce();
		loop_rate.sleep();

	}

	return 0;
}