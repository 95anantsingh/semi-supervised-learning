
var img_dir = '';
var dataset = "yahoo_answers";
var strategy = "pc_top_k";
var strat_val = "100";
var run = 'Run1'
var sampling = "equal";
var labeled_size = "100";
var unlabeled_ratio = "50"
var unlabeled_size = "15000";
    
var padding = 0;
var total_width3 = 0;
var total_img_width3 = 0;
var px_val3 = 0

var img_1 = document.getElementById("image_1")
var img_2 = document.getElementById("image_2")
var img_3 = document.getElementById("image_3")
var img_4 = document.getElementById("image_4")
var img_5 = document.getElementById("image_5")
var img_6 = document.getElementById("image_6")
var img_7 = document.getElementById("image_7")
var img_8 = document.getElementById("image_8")
var img_9 = document.getElementById("image_9")
var img_container = document.getElementById("image_container");

var dataset_selector = document.getElementById("dataset_selector")
var strategy_selector = document.getElementById("strategy_selector")
var run_selector = document.getElementById("run_selector")
var labeled_selector = document.getElementById("labeled_selector")
var unlabeled_size_selector = document.getElementById("unlabeled_size_selector")
var unlabeled_ratio_selector = document.getElementById("unlabeled_ratio_selector")


function refresh_widths(){
    // var total_width1 = window.innerWidth-(2*padding)-120;
    
    padding = window.innerWidth - img_container.clientWidth;
    total_width3 = window.innerWidth-(2*padding)+15;
    total_img_width3 = 0;
    px_val3 = 0

    // var total_img_width1 = img_1.naturalWidth + img_2.naturalWidth;
    // var total_img_width2 = img_3.naturalWidth + img_4.naturalWidth;
    total_img_width3 = img_5.naturalWidth + img_6.naturalWidth + img_7.naturalWidth + img_8.naturalWidth + img_9.naturalWidth;

    // var px_val1 = total_width1/total_img_width1;
    // img_1.width = Math.min(img_1.naturalWidth * px_val1,img_1.naturalWidth);
    // img_2.width = Math.min(img_2.naturalWidth * px_val1,img_2.naturalWidth);

    // var px_val2 = total_width1/total_img_width2;
    // img_3.width = Math.min(img_3.naturalWidth * px_val2,img_3.naturalWidth);
    // img_4.width = Math.min(img_4.naturalWidth * px_val2,img_4.naturalWidth);

    px_val3 = total_width3/total_img_width3;
    img_5.width = Math.min(img_5.naturalWidth * px_val3,img_5.naturalWidth);
    img_6.width = Math.min(img_6.naturalWidth * px_val3,img_6.naturalWidth);
    img_7.width = Math.min(img_7.naturalWidth * px_val3,img_7.naturalWidth);
    img_8.width = Math.min(img_8.naturalWidth * px_val3,img_8.naturalWidth);
    img_9.width = Math.min(img_9.naturalWidth * px_val3,img_9.naturalWidth);
}

function dataset_change() {
    // dataset = dataset_selector.options[dataset_selector.selectedIndex].value;
    // if(dataset=='dbpedia_14'){
    //     labeled_selector.innerHTML='\
    //         <option value="7_35">Class: 7, Size: 35</option>\
    //         <option value="7_70">Class: 7, Size: 70</option>\
    //         <option value="7_140">Class: 7, Size: 140</option>\
    //         <option value="7_350">Class: 7, Size: 350</option>\
    //         <option value="7_1050">Class: 7, Size: 1050</option>';
    //     unlabeled_selector.innerHTML='\
    //         <option value="7_5250">Class: 7, Size: 5250</option>\
    //         <option value="7_10500">Class: 7, Size: 10500</option>\
    //         <option value="14_10500">Class: 14, Size: 10500</option>\
    //         <option value="14_21000">Class: 14, Size: 21000</option>';
    // }
    // else{
    //     labeled_selector.innerHTML='\
    //         <option value="5_25">Class: 5, Size: 25</option>\
    //         <option value="5_50">Class: 5, Size: 50</option>\
    //         <option value="5_100">Class: 5, Size: 100</option>\
    //         <option value="5_250">Class: 5, Size: 250</option>\
    //         <option value="5_750">Class: 5, Size: 750</option>';
    //     unlabeled_selector.innerHTML='\
    //         <option value="5_5000">Class: 5, Size: 5000</option>\
    //         <option value="5_10000">Class: 5, Size: 10000</option>\
    //         <option value="10_10000">Class: 10, Size: 10000</option>\
    //         <option value="10_20000">Class: 10, Size: 20000</option>';
    // }
    strategy_change();
    // refresh_data();
}

function strategy_change() {
    // strategy = strategy_selector.options[strategy_selector.selectedIndex].value;
    // dataset = dataset_selector.options[dataset_selector.selectedIndex].value;
    
    // if (strategy=="pc_top_k"){
    //     if(dataset=='dbpedia_14'){
    //         strat_val_selector.innerHTML='\
    //             <option value="35">35</option>\
    //             <option value="70">70</option>';
    //     }
    //     else{
    //         strat_val_selector.innerHTML='\
    //             <option value="50">50</option>\
    //             <option value="100">100</option>';
    //     }
    // }
    // else{
    //     strat_val_selector.innerHTML='\
    //         <option value="250">250</option>\
    //         <option value="500">500</option>';
    // }
    refresh_data();
}

function refresh_data(){
    dataset = dataset_selector.options[dataset_selector.selectedIndex].value;
    strategy = strategy_selector.options[strategy_selector.selectedIndex].value.split("_");
    run = run_selector.options[run_selector.selectedIndex].value;
    labeled_size = labeled_selector.options[labeled_selector.selectedIndex].value;
    unlabeled_size = unlabeled_size_selector.options[unlabeled_size_selector.selectedIndex].value;
    unlabeled_ratio = unlabeled_ratio_selector.options[unlabeled_ratio_selector.selectedIndex].value;

    strat_val = strategy.pop()
    sampling = strategy.pop()
    strategy = strategy.join('_')
    update_images();
}

function update_images(){
    img_dir="visuals/v3.1/"+dataset+"/"+strategy+"/"+run+"/"+strat_val+"_"+sampling+"_"+labeled_size+"_"+unlabeled_size+"_"+unlabeled_ratio;
    img_1.src = img_dir+"/acc_plot.png";
    img_2.src = img_dir+"/cw_acc_plot.png";
    img_3.src = img_dir+"/selection_plot.png";
    img_4.src = img_dir+"/cw_f1_plot.png";
    img_5.src = img_dir+"/predicted_set_class_dist.png";
    img_6.src = img_dir+"/predicted_prob_dist.png";
    img_7.src = img_dir+"/pseudo_set_class_dist.png";
    img_8.src = img_dir+"/selected_prob_dist.png";
    img_9.src = img_dir+"/selected_set_class_dist.png";

    setTimeout(refresh_widths, 2000);
}

function console_log(message){
    console.log(message)
}


// Zoom Handling
// var header = document.getElementById("header");
// var header_text = document.getElementById("header_text");
// var header_link = document.getElementById("header_link");
//
// org_header_height=header.clientHeight;
// org_header_text_height=header_text.clientHeight;
// org_header_link_height=header_link.clientHeight;
//
// org_screen_height= window.innerHeight;
// window.addEventListener("resize", getSizes, false)
//
// function getSizes(event){
//     let body = document.body
//     body.width = window.innerWidth
//     body.height = window.innerHeight
//     header.style.height=(org_header_height*window.innerHeight/org_screen_height)+"px"
//     // header_text.style.height=(org_header_text_height*window.innerHeight/org_screen_height)+"px"
//     // header_link.style.height=(org_header_link_height*window.innerHeight/org_screen_height)+"px"
//     console_log(body.width +"px x "+ body.height + "px"+ header.clientHeight)
//   }
// getSizes()