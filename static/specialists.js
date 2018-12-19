const IMAGE_SIZE = 224;
const COORD_SCALE = 5.0;
let scores1 = [];
let scores2 = [];
let scoresPred = [];
let vhsModel = null;

	loadModel();
$(document).ready(function() {
})

async function loadModel() {
	vhsModel = await tf.loadModel('models/vhs/model.json');

	$.getJSON('data/specialist_data.json', function(data) {
		fillTable(data);
	})
}

function getVHS(points) {
	let minorAxis = distance(points[4], points[5], points[6], points[7]);
	let majorAxis = distance(points[0], points[1], points[2], points[3]);
	let vertebraeLine = distance(points[8], points[9], points[16], points[17]);
	let vhs = (minorAxis+majorAxis)/(vertebraeLine*0.25);

	return vhs;
}

function distance(x1, y1, x2, y2) {
	let x = x2 - x1;
	let y = y2 - y1;
	return Math.sqrt(x*x + y*y);
}

function drawVhs(ctx, points, x, y, pred=true) {

	let vhs = getVHS(points)


	let text = ''
	if(pred) {
		text = 'Auto VHS: '+vhs.toFixed(2);
	} else {
		text = 'GT VHS: '+vhs.toFixed(2);
	}
	ctx.fillStyle = 'rgb(250, 250, 0)';
	ctx.fillText(text, 5, 15);
}

function drawLine(ctx, x1, y1, x2, y2) {
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
}

function drawCanvas(canvas, image, points, pred=true) {
	let ctx = canvas.getContext('2d');
	let w = image.width;
	let h = image.height;
	canvas.width = w;
	canvas.height = h;

	ctx.drawImage(image, 0, 0, w, h);

	ctx.strokeStyle = 'rgb(200,150,0)'

	drawLine(ctx, points[0]*w, points[1]*h, points[2]*w, points[3]*h);
	drawLine(ctx, points[4]*w, points[5]*h, points[6]*w, points[7]*h);
	drawLine(ctx, points[8]*w, points[9]*h, points[16]*w, points[17]*h);

	drawVhs(ctx, points, points[16]*w+20, points[17]*h, pred);

	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[0]*w, points[1]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[2]*w, points[3]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[4]*w, points[5]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[6]*w, points[7]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[8]*w, points[9]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[10]*w, points[11]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[12]*w, points[13]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[14]*w, points[15]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(points[16]*w, points[17]*h, 2, 2);
}

async function predict(image) {

	const logits = tf.tidy(() => {
		const img = tf.fromPixels(image, 1).toFloat();
		const offset = tf.scalar(255);
		const normalized = tf.image.resizeBilinear(img.div(offset), [IMAGE_SIZE, IMAGE_SIZE]);
		const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);

		return vhsModel.predict(batched);
	});

	const values = await logits.data();
	return values
}

function fillRow(obj) {

	var tableBody = document.getElementById('cardio-table').getElementsByTagName('tbody')[0];
	var row = tableBody.insertRow(tableBody.rows.length);

	let image = new Image();
	image.crossOrigin = 'anonymous';
	image.row = row;
	image.width = IMAGE_SIZE*2;
	image.height = IMAGE_SIZE*2;
	image.style.width=IMAGE_SIZE*2 + 'px';
	image.style.height=IMAGE_SIZE*2 + 'px';
	image.src = obj.url;
	image.onload = async function() {
		var canvas1 = document.createElement('canvas');
		var canvas2 = document.createElement('canvas');
		var canvasPred = document.createElement('canvas');

		var cell1 = this.row.insertCell(0);
		cell1.style.width = '500px';
		drawCanvas(canvas1, this, obj.annotation_1, pred=false);
		cell1.appendChild(canvas1);

		var cell2 = this.row.insertCell(1);
		cell2.style.width = '500px';
		drawCanvas(canvas2, this, obj.annotation_2, pred=false);
		cell2.appendChild(canvas2);

		let preds = await predict(this);

		for (let i=0;i<preds.length;i++) {
			preds[i] = preds[i] / COORD_SCALE;
		}

		scoresPred.push(getVHS(preds));

		cellPred = this.row.insertCell(2);
		cellPred.style.width = '500px';
		drawCanvas(canvasPred, this, preds);
		cellPred.appendChild(canvasPred);

	}

}

function get_mse(x, y) {

	
	
	let mean_error = 0;
	for (let i = 0; i < x.length; i++) {
		mean_error = mean_error + (x[i]-y[i])*(x[i]-y[i]);
	}

	mean_error = mean_error / x.length;

	return mean_error;
}


function fillTable(data) {

	for (let i=0; i < data.length; i++) {
		var obj = data[i];
		scores1.push(getVHS(obj.annotation_1));
		scores2.push(getVHS(obj.annotation_2));
		fillRow(obj);
	}
}

setInterval(function() {

  if(scores1.length === scoresPred.length) {
  	
  	$('#mseCardio1').html(get_mse(scores1, scoresPred).toFixed(3));
  	$('#mseCardio2').html(get_mse(scores2, scoresPred).toFixed(3));
  	$('#mseCardioCardio').html(get_mse(scores1, scores2).toFixed(3));

  }
}, 5000);