#include "ros/ros.h"
#include <sstream>
#include <bits/stdc++.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <cassert>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace message_filters;

static ros::Publisher ir_left_pub;
static ros::Publisher ir_right_pub;
static ros::Publisher rgb_fl_pub;
static ros::Publisher rgb_fr_pub;
static ros::Publisher rgb_bl_pub;
static ros::Publisher rgb_br_pub;

static int counter = 0;
static ros::Time start_time;
static double burst_period = 1.0;

const int burst_img_count = 5;

void callback(const sensor_msgs::Image::ConstPtr& ir_left,
              const sensor_msgs::Image::ConstPtr& ir_right,
              const sensor_msgs::Image::ConstPtr& rgb_fl,
              const sensor_msgs::Image::ConstPtr& rgb_fr,
              const sensor_msgs::Image::ConstPtr& rgb_bl,
              const sensor_msgs::Image::ConstPtr& rgb_br) {

    ros::Duration delayed_time = ros::Time::now() - start_time;

    if (delayed_time.toSec() > burst_period){

        // preprocess images
        if(counter < burst_img_count){
            ir_left_pub.publish(ir_left);
            ir_right_pub.publish(ir_right);
            rgb_fl_pub.publish(rgb_fl);
            rgb_fr_pub.publish(rgb_fr);
            rgb_bl_pub.publish(rgb_bl);
            rgb_br_pub.publish(rgb_br);

            counter++;
        }else{
            start_time = ros::Time::now();
            counter = 0;
        }
    }

}


int main(int argc, char** argv) {

    ros::init(argc, argv, "ImageSampler");
	ros::NodeHandle nh;

	// Subscribe to image topics
    message_filters::Subscriber<sensor_msgs::Image> ir_1_sub(nh, "flir_boson_left/image_raw_16", 1);
    message_filters::Subscriber<sensor_msgs::Image> ir_2_sub(nh, "flir_boson_right/image_raw_16", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_1_sub(nh, "/blackfly/camera_front_left/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_2_sub(nh, "/blackfly/camera_front_right/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_3_sub(nh, "/blackfly/camera_back_left/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_4_sub(nh, "/blackfly/camera_back_right/image_color", 1);

	// Publish image bursts in new topic
    ir_left_pub = nh.advertise<sensor_msgs::Image>("/ir_left_burst", 1000);
    ir_right_pub = nh.advertise<sensor_msgs::Image>("/ir_right_burst", 1000);
    rgb_fl_pub = nh.advertise<sensor_msgs::Image>("/rgb_fl_burst", 1000);
    rgb_fr_pub = nh.advertise<sensor_msgs::Image>("/rgb_fr_burst", 1000);
    rgb_bl_pub = nh.advertise<sensor_msgs::Image>("/rgb_bl_burst", 1000);
    rgb_br_pub = nh.advertise<sensor_msgs::Image>("/rgb_br_burst", 1000);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), ir_1_sub, ir_2_sub, rgb_1_sub, rgb_2_sub, rgb_3_sub, rgb_4_sub);

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6));
	
    start_time = ros::Time::now();
	ros::spin();

	return 0;

}
