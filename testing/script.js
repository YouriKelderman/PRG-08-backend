document.addEventListener('DOMContentLoaded', init);

let testButton = document.getElementById('testButton')
let calculateTable = document.getElementById('fillTable')

let topRow = document.getElementById('topRow')
let table = document.getElementById('table')

let testText = document.getElementById('test')
let accuracyText = document.getElementById('accuracy')
let timesText = document.getElementById('times');

let testRunning = false;
let intervalId;

let train;
let test;

let accuracy = 0;
let totalTestPredictions = 0;
let correctPredictions = 0;

let importedTrainingData;

const nn = ml5.neuralNetwork({task: 'classification', debug: true})
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}

function init() {
    nn.load(modelDetails, () => console.log("het model is geladen!"))

    testButton.addEventListener('click', function () {
        if (testRunning === false) {
            intervalId = setInterval(() => {
                startTesting();
            }, 50);
            testRunning = true
            testButton.innerText = 'Stop Testing'
        } else {
            clearInterval(intervalId);
            testRunning = false;
            testButton.innerText = 'Start Testing'
        }
    });

    calculateTable.addEventListener('click', fillTable);

    fetchData();
}

async function startTesting() {
    if (!importedTrainingData) {
        console.error('Training data not fetched yet.');
        return;
    }

    const testpose = importedTrainingData[Math.floor(Math.random() * importedTrainingData.length)]; // Use Math.floor to ensure index is an integer
    const prediction = await nn.classify(testpose.pose);
    totalTestPredictions++

    if (prediction[0].label === testpose.label) {
        correctPredictions++

        testText.style.color = 'green'
    } else {
        testText.style.color = 'red'
    }

    accuracy = correctPredictions / totalTestPredictions;

    testText.innerHTML = `Voorspelling: <b>${prediction[0].label}</b>. Correct: <b>${testpose.label}</b>`

    timesText.innerHTML = "Doorlopen poses:" + totalTestPredictions;
    accuracyText.innerText = accuracy
}

async function fetchData() {
    try {
        const response = await fetch('trainingData.json'); // Fetch pointing.json
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        importedTrainingData = await response.json();
        console.log(importedTrainingData);
        importedTrainingData = importedTrainingData.toSorted(() => (Math.random() - 0.5))

        train = importedTrainingData.slice(0, Math.floor(importedTrainingData.length * 0.8))
        test = importedTrainingData.slice(Math.floor(importedTrainingData.length * 0.8) + 1)

    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

async function fillTable() {
    const uniqueLabels = [...new Set(importedTrainingData.map(data => data.label))];
    console.log(uniqueLabels);

    table.innerHTML = '';

    let topRow = document.createElement('tr');
    let emptyHeader = document.createElement('th');
    topRow.appendChild(emptyHeader); // Empty cell for spacing

    for (let i = 0; i < uniqueLabels.length; i++) {
        let th = document.createElement('th');
        th.innerText = uniqueLabels[i];
        topRow.appendChild(th);
    }
    table.appendChild(topRow);

    const confusionMatrix = {};

    for (let trueLabel of uniqueLabels) {
        confusionMatrix[trueLabel] = {};
        for (let predictedLabel of uniqueLabels) {
            confusionMatrix[trueLabel][predictedLabel] = 0;
        }
    }

    for (let data of importedTrainingData) {
        const trueLabel = data.label;
        const predictedLabel = await predictLabel(data.pose);

        if (trueLabel === predictedLabel) {
            confusionMatrix[trueLabel][predictedLabel]++;
        } else {
            confusionMatrix[trueLabel][predictedLabel]++;
        }
    }


    for (let trueLabel of uniqueLabels) {
        let tr = document.createElement('tr');
        let th = document.createElement('th');
        th.innerText = trueLabel;
        tr.appendChild(th);

        for (let predictedLabel of uniqueLabels) {
            let td = document.createElement('td');
            td.innerText = confusionMatrix[trueLabel][predictedLabel];

            if (trueLabel === predictedLabel) {
                td.style.background = 'lightgreen'
            } else {
                if (confusionMatrix[trueLabel][predictedLabel] !== 0) {
                    td.style.background = 'lightcoral'
                }
            }

            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
}

async function predictLabel(pose) {
    const prediction = await nn.classify(pose);
    return prediction[0].label;
}