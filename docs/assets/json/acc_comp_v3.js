
var dataset = "dbpedia_14";
var strategy = "top_k";
var sampling = "";

var plotly_div = document.getElementById('plotly_div');
var plotly_iframe = document.getElementById('plotly_iframe');
var dataset_selector = document.getElementById("dataset_selector");
var strategy_selector = document.getElementById("strategy_selector");
var sampling_selector = document.getElementById("sampling_selector");
var header = document.getElementById("header_");
var selectors = document.getElementById("selectors");
var maximize_button = document.getElementById('maximize_button');

function strategy_change(){
    // strategy = strategy_selector.options[strategy_selector.selectedIndex].value;
    // if (strategy==''){
    //     sampling_selector.disabled=true;
    //     sampling_selector.innerHTML='\
    //         <option value="" selected>All</option>\
    //         <option value="">Equal</option>\
    //         <option value="random_">Random</option>'
    // }
    // else{
    //     sampling_selector.disabled=false;
    //     sampling_selector.innerHTML='\
    //         <option value="">Equal</option>\
    //         <option value="random_">Random</option>'
    // }
    refresh_data();
}

function refresh_data(){
    // dataset = dataset_selector.options[dataset_selector.selectedIndex].value;
    // strategy = strategy_selector.options[strategy_selector.selectedIndex].value;
    // sampling = sampling_selector.options[sampling_selector.selectedIndex].value;
    update_chart();
}

var plot_dir = "";
var data = '';
var padding = 0;
function update_chart(){
    // // iframe_height = window.innerHeight;
    // plot="visuals/v2.0/"+dataset+"/"+sampling+strategy+'/acc_comparison.html'; 
    // // plotly_html.style.height=iframe_height+'px';  
    // plotly_iframe.src= plot;
    // // plotly_html.width=screen.width;
}


// Maximize plot
var scrollTop = 0;
var scrollLeft = 0;
var maximized = false;
function maximize() {
    if (maximized){
        maximize_button.style.backgroundColor='white';
        maximize_button.style.color='black';
        plotly_div.classList.remove('maximize');
        window.scrollTo(scrollLeft, scrollTop);
        maximized = false;
    }
    else{
        maximize_button.style.backgroundColor='indianred';
        maximize_button.style.color='white';
        scrollTop =  window.pageYOffset || document.documentElement.scrollTop;
        scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        plotly_div.classList.add('maximize');
        plotly_iframe.style.height=(window.innerHeight-5)+'px';
        maximized = true;
    }
}

// refresh_data();


// Style iframe
function style_iframe(){
    iframe = window.frames[0].document;
    // iframe.body.style.margin = 0;
    // iframe.svg.setAttribute('width','100%');
}
