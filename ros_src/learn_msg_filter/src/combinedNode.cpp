#include <memory>
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

#include <image_transport/image_transport.h>


using namespace message_filters;

//ros::Publisher ir_left_pub;
//ros::Publisher ir_right_pub;
//ros::Publisher rgb_fl_pub;
static image_transport::Publisher ir_left_pub;
static image_transport::Publisher ir_right_pub;
static image_transport::Publisher rgb_fl_pub;
static image_transport::Publisher rgb_fr_pub;

static int counter = 0;
static ros::Time start_time;

using namespace cv;
using namespace std;



void callback(const sensor_msgs::Image::ConstPtr& n1,
              const sensor_msgs::Image::ConstPtr& n2,
              const sensor_msgs::Image::ConstPtr& n3,
              const sensor_msgs::Image::ConstPtr& n4) {

    ros::Duration delayed_time = ros::Time::now() - start_time;
    std::cout << "Images sync.... " << std::endl;

    if (true){

        // preprocess images
        if(true){

            cv_bridge::CvImagePtr ir_1_mat_ptr = cv_bridge::toCvCopy(*n1, "16UC1");
            cv_bridge::CvImagePtr ir_2_mat_ptr = cv_bridge::toCvCopy(*n2, "16UC1");
            cv_bridge::CvImagePtr rgb_mat_ptr1 = cv_bridge::toCvCopy(*n3, "mono8");
            cv_bridge::CvImagePtr rgb_mat_ptr2 = cv_bridge::toCvCopy(*n4, "mono8");

            //convert IR 16bit to IR 8bit



            cv::Mat copy1;
            cv::Mat copy2;

            ir_1_mat_ptr->image.convertTo(copy1, CV_32FC1);
            ir_2_mat_ptr->image.convertTo(copy2, CV_32FC1);

            cv::threshold(copy1, copy1, 30000, 1, cv::THRESH_TRUNC);
            cv::threshold(copy2, copy2, 30000, 1, cv::THRESH_TRUNC);

            copy1.convertTo(ir_1_mat_ptr->image, CV_16UC1);
            copy2.convertTo(ir_2_mat_ptr->image, CV_16UC1);



            cv::Mat copy;

            cv::normalize(ir_1_mat_ptr->image, ir_1_mat_ptr->image, 0, 255, cv::NORM_MINMAX);
            cv::normalize(ir_2_mat_ptr->image, ir_2_mat_ptr->image,  0, 255, cv::NORM_MINMAX);

            ir_1_mat_ptr->image.convertTo(ir_1_mat_ptr->image, CV_8UC1);
            ir_2_mat_ptr->image.convertTo(ir_2_mat_ptr->image, CV_8UC1);

            // invert IR and RGB
            cv::Mat inverted_ir_1;
            cv::Mat inverted_ir_2;
            cv::Mat inverted_rgb;
            cv::Mat inverted_rgb2;

//            cv::threshold(ir_1_mat_ptr->image, ir_1_mat_ptr->image, 240, 255, cv::THRESH_TRUNC);
//            cv::threshold(ir_2_mat_ptr->image, ir_2_mat_ptr->image, 240, 255, cv::THRESH_TRUNC);



            cv::subtract(cv::Scalar::all(255),ir_1_mat_ptr->image,inverted_ir_1);
            cv::subtract(cv::Scalar::all(255),ir_2_mat_ptr->image,inverted_ir_2);
            cv::subtract(cv::Scalar::all(255),rgb_mat_ptr1->image,inverted_rgb);
            cv::subtract(cv::Scalar::all(255),rgb_mat_ptr2->image,inverted_rgb2);

//            cv::threshold(inverted_ir_1, inverted_ir_1, 230, 255, cv::THRESH_TRUNC);
//            cv::threshold(inverted_ir_2, inverted_ir_2, 230, 255, cv::THRESH_TRUNC);

            //cv::normalize(inverted_ir_1, inverted_ir_1, 0, 255, cv::NORM_MINMAX);
             //cv::normalize(inverted_ir_2, inverted_ir_2,  0, 255, cv::NORM_MINMAX);

            cv::threshold(inverted_ir_1, inverted_ir_1, 100, 255, cv::THRESH_BINARY);
            cv::threshold(inverted_ir_2, inverted_ir_2, 100, 255, cv::THRESH_BINARY);

            ir_1_mat_ptr->image = inverted_ir_1;
            ir_2_mat_ptr->image = inverted_ir_2;

            cv::threshold(inverted_rgb, inverted_rgb, 140, 255, cv::THRESH_BINARY);
            cv::threshold(inverted_rgb2, inverted_rgb2, 140, 255, cv::THRESH_BINARY);



            ir_1_mat_ptr->encoding = "mono8";
            ir_2_mat_ptr->encoding = "mono8";
            rgb_mat_ptr1->image = inverted_rgb;
            rgb_mat_ptr2->image = inverted_rgb2;


            ir_left_pub.publish(ir_1_mat_ptr->toImageMsg());
            ir_right_pub.publish(ir_2_mat_ptr->toImageMsg());
            rgb_fl_pub.publish(rgb_mat_ptr1->toImageMsg());
            rgb_fr_pub.publish(rgb_mat_ptr2->toImageMsg());

            counter++;
        }else{
            start_time = ros::Time::now();
            counter = 0;
        }
    }

}


int main(int argc, char** argv) {

    ros::init(argc, argv, "DataSelector");
	ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

	// Subscribe to image topics
    message_filters::Subscriber<sensor_msgs::Image> f_sub(nh, "/ir_left_burst", 1);
    message_filters::Subscriber<sensor_msgs::Image> f2_sub(nh, "/ir_right_burst", 1);
    message_filters::Subscriber<sensor_msgs::Image> s1_sub(nh, "/rgb_fl_burst", 1);
    message_filters::Subscriber<sensor_msgs::Image> s2_sub(nh, "/rgb_fr_burst", 1);

	// Publish image bursts in new topic
    ir_left_pub = it.advertise("ir_left_post", 1);
    ir_right_pub = it.advertise("ir_right_post", 1);
    rgb_fl_pub = it.advertise("rgb_left_post", 1);
    rgb_fr_pub = it.advertise("rgb_right_post", 1);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), f_sub, f2_sub, s1_sub, s2_sub);

    // callback is called every time, two IR images and a RGB image arrive together (every other IR image as it has doubled frequency)
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));
	
    start_time = ros::Time::now();
	ros::spin();

	return 0;

}
