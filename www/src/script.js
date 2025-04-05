import { select } from 'd3-selection';
import { hierarchy, tree } from 'd3-hierarchy';
import { linkHorizontal } from 'd3-shape';

if (module.hot) {
    module.hot.accept();
  }

// Sample data mimicking anytree JSON output
const data = {
  "name": "Birthday Party Task",
  "reasoning": "",
  "children": [
    {
      "name": "Design the Garden Layout",
      "reasoning": "A well-thought-out plan is needed...",
      "children": [
        {
          "name": "NO SUBDIVISION",
          "reasoning": "Garden layout design complexity is unknown."
        }
      ]
    },
    {
      "name": "Choose Plants",
      "reasoning": "Selecting the right plants is crucial...",
      "children": [
        {
          "name": "NO SUBDIVISION",
          "reasoning": "No time frame or plant type specified."
        }
      ]
    },
    {
      "name": "Prepare the Soil",
      "reasoning": "Proper soil preparation ensures nutrient-rich growth.",
      "children": []
    }
  ]
};

// Convert the nested JSON into a d3 hierarchy.
const root = hierarchy(data);

// Create a tree layout with specified dimensions.
const treeLayout = tree().size([600, 900]);
treeLayout(root);

// Select the SVG element and create a group with margins.
const svg = select("svg");
const g = svg.append("g")
             .attr("transform", "translate(40,40)");

// Render links between nodes.
const linkGenerator = linkHorizontal()
                        .x(d => d.y)
                        .y(d => d.x);

g.selectAll(".link")
  .data(root.links())
  .enter()
  .append("path")
  .attr("class", "link")
  .attr("d", linkGenerator);

// Render nodes.
const node = g.selectAll(".node")
              .data(root.descendants())
              .enter()
              .append("g")
              .attr("class", "node")
              .attr("transform", d => `translate(${d.y},${d.x})`);

// Append a circle to each node.
node.append("circle")
    .attr("r", 5);

// Append text to each node, displaying the node's name.
node.append("text")
    .attr("dy", 3)
    .attr("x", d => d.children ? -10 : 10)
    .style("text-anchor", d => d.children ? "end" : "start")
    .text(d => d.data.name);
