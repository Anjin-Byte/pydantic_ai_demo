import { select } from 'd3-selection';
import { hierarchy, tree } from 'd3-hierarchy';
import { linkHorizontal } from 'd3-shape';

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
            console.log("Loaded JSON:", data);
            renderTree(root, svg)
        })
        .catch((error) => {
            console.error("Error parsing JSON:", error);
        });
});

function renderTree(root, svg) {
    const treeLayout = tree().size([600, 900]);
    treeLayout(root);

    const g = svg.append("g")
        .attr("transform", "translate(40,40)");

    const linkGenerator = linkHorizontal()
        .x(d => d.y)
        .y(d => d.x);

    g.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", linkGenerator);

    const node = g.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${d.y},${d.x})`);

    node.append("circle")
        .attr("r", 5);

    node.append("text")
        .attr("dy", 3)
        .attr("x", d => d.children ? -10 : 10)
        .style("text-anchor", d => d.children ? "end" : "start")
        .text(d => d.data.name);
}

renderTree(root, svg)