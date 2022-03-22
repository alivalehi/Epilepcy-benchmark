function GetSelectedTextValue(ddlFruits) {
    var selectedText = ddlFruits.options[ddlFruits.selectedIndex].innerHTML;
    var selectedValue = ddlFruits.value;
    // alert("Selected Text: " + selectedText + " Value: " + selectedValue);
    trace2 = trace(models[selectedValue][0],models[selectedValue][1],models[selectedValue][2]); 
    data = [trace1, trace2]
    Plotly.newPlot('myDiv', data, layout); 
}

var trace = (modelName, a,b) =>
{
    return {
    x: ['Sensitivity', 'Specificity'],
    y: [a, b],
    name: modelName,
    type: 'bar',
       marker: {
        color: 'red'
      }
    }
};



var trace1 = {
    x: ['Sensitivity', 'Specificity'],
    y: [20, 14],
    name: 'The propose Model',
    type: 'bar',
     marker: {
        color: 'green'
      }
  };
  
  var trace2 = trace("model1",0,0);  
  var data = [trace1, trace2];
  
  var layout = {barmode: 'group'};
  
  Plotly.newPlot('myDiv', data, layout);

let models = [["model1",20,40],["model2",30,60]]

  