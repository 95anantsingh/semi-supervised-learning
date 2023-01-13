
var dataset = "dbpedia_14";
var strategy = "top_k";
var sampling = "";
var plotly_html = document.getElementById('plotly_html');
var dataset_selector = document.getElementById("dataset_selector");
var strategy_selector = document.getElementById("strategy_selector");
var sampling_selector = document.getElementById("sampling_selector");
var header = document.getElementById("header_");
var selectors = document.getElementById("selectors");


function strategy_change(){
    strategy = strategy_selector.options[strategy_selector.selectedIndex].value;
    if (strategy==''){
        sampling_selector.innerHTML='\
            <option value="" selected>select</option>\
            <option value="">Equal</option>\
            <option value="random_">Random</option>'
    }
    else{
        sampling_selector.innerHTML='\
            <option value="">Equal</option>\
            <option value="random_">Random</option>'
    }
    refresh_data();
}

function refresh_data(){
    dataset = dataset_selector.options[dataset_selector.selectedIndex].value;
    strategy = strategy_selector.options[strategy_selector.selectedIndex].value;
    sampling = sampling_selector.options[sampling_selector.selectedIndex].value;
    update_chart();
}

var plot_dir = "";
var data = '';
var padding = 0;
function update_chart(){
    iframe_height = window.innerHeight;
    plot="visuals/"+dataset+"/"+sampling+strategy+'/acc_comparison.html'; 
    plotly_html.style.height=iframe_height+'px';  
    plotly_html.src= plot;
}

refresh_data();