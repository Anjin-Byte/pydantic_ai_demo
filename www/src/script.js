import { select } from 'd3-selection';
import { hierarchy, tree } from 'd3-hierarchy';
import { linkHorizontal, linkVertical } from 'd3-shape';
import { groups } from 'd3-array'

if (module.hot) {
    module.hot.accept();
}
class JsonDataHandler {
    constructor() {
        this.data = null;
    }

    loadFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    this.data = JSON.parse(e.target.result);
                    resolve(this.data);
                } catch (err) {
                    reject(new Error("Error parsing JSON: " + err));
                }
            };
            reader.onerror = () => {
                reject(new Error("Error reading file"));
            };
            reader.readAsText(file);
        });
    }

    getData() {
        return this.data;
    }

    hasData() {
        return this.data !== null;
    }

    clearData() {
        this.data = null;
    }
}

const jsonHandler = new JsonDataHandler();
const svg = select("svg");
let root = hierarchy('');

document.getElementById("loadButton").addEventListener("click", () => {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    jsonHandler.loadFile(file)
        .then((data) => {
            root = hierarchy(jsonHandler.getData());
            //console.log("Loaded JSON:", data);
            renderTree(root, svg)
        })
        .catch((error) => {
            console.error("Error parsing JSON:", error);
        });
});

const customLinkGenerator = d => {
  // Use the same coordinates as in the node transform.
  const sourceX = d.source.x;
  const sourceY = d.source.y + (d.source.sinkingOffset || 0);
  const targetX = d.target.x;
  const targetY = d.target.y + (d.target.sinkingOffset || 0);

  // Compute a midpoint along the horizontal axis (x) for a smooth curve.
  const midX = (sourceX + targetX) / 2;

  // Construct a cubic Bézier curve:
  // Start at source, control points at (midX, sourceY) and (midX, targetY), end at target.
  return `M${sourceX},${sourceY}
          C${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
};

function renderTree(root, svg) {
    const treeLayout = tree().size([2000, 3000]);
    treeLayout(root);

    const g = svg.append("g")
    //.attr("transform", "translate(40,40)");




    // Get all nodes.
    const nodes = root.descendants();

    // Group nodes by their depth (i.e., row in the tree).
    // d3.groups creates an array of [key, values] pairs.
    const nodesByDepth = groups(nodes, d => d.depth);

    // Define a unit offset—this might be related to your font size or node size.
    const unitOffset = 22; // for example, 10 pixels

    // For each group of nodes at the same depth, assign an extra offset.
    nodesByDepth.forEach(([depth, nodesAtDepth]) => {
        nodesAtDepth.forEach((node, i) => {
            // Each node gets a sinking offset based on its order at that depth.
            node.sinkingOffset = i * unitOffset;
            //console.log(node)
        });
    });


    const node = g.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", "node")
        //.attr("transform", d => `translate(${d.y},${d.x})`);
        //.attr("transform", d => `translate(${d.x}, ${d.y})`);
        .attr("transform", d => `translate(${d.x}, ${d.y + (d.sinkingOffset || 0)})`);

    const linkGenerator = linkVertical()
        .x(d => d.x)  // horizontal coordinate comes from d.y
        .y(d => d.y); // vertical coordinate comes from d.x

    node.append("circle")
        .attr("r", 5);

    node.append("text")
        .attr("text-anchor", "middle")  // centers text horizontally
        .attr("dy", "1.2em")              // moves text below the node (adjust as needed)
        .text(d => d.data.name);

    g.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", customLinkGenerator);

    node.each(function () {
        const g = select(this);
        const text_element = g.select("text").node();
        const bbox = text_element.getBBox();
        const padding = 4;

        g.insert("rect", "text.node-label")
            .attr("x", bbox.x - padding)
            .attr("y", bbox.y - padding)
            .attr("width", bbox.width + 2 * padding)
            .attr("height", bbox.height + 2 * padding)
            .style("fill", "lightsteelblue")
            .style("stroke", "steelblue");
    });

    node.append("text")
        .attr("text-anchor", "middle")  
        .attr("dy", "1.2em")  
        .text(d => d.data.name);

}

renderTree(root, svg)