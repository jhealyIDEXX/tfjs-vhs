const POINT_LABELS = {
	0: 'Apex x',
	1: 'Apex y',
	2: 'TC x',
	3: 'TC y',
	4: 'M1 x',
	5: 'M1 y',
	6: 'M2 x',
	7: 'M2 y',
	8: 'T4 x',
	9: 'T4 y',
	10: 'T5 x',
	11: 'T5 y',
	12: 'T6 x',
	13: 'T6 y',
	14: 'T7 x',
	15: 'T7 y',
	16: 'T8 x',
	17: 'T8 y'  	
};
Object.freeze(POINT_LABELS)
const IMAGE_SIZE = 224;
const COORD_SCALE = 5.0;
var vhsModel = null;
let gt_scores = [];
let pred_scores = [];



$(document).ready(function() {
	$(function() {
		$( "points_dialog" ).dialog({
			autoOpen: false
		});
	});

	loadModel();
});


async function loadModel() {
	console.log('model loaded');
	vhsModel =  await tf.loadModel('models/vhs/model.json');

	$.getJSON('data/eval_data.json', function(data) {
		fillTable(data);
	});
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

function drawLine(ctx, x1, y1, x2, y2) {
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
}

function distance(x1, y1, x2, y2) {
	let x = x2 - x1;
	let y = y2 - y1;
	return Math.sqrt(x*x + y*y);
}

function getVHS(points) {
	let minorAxis = distance(points[4], points[5], points[6], points[7]);
	let majorAxis = distance(points[0], points[1], points[2], points[3]);
	let vertebraeLine = distance(points[8], points[9], points[16], points[17]);
	let vhs = (minorAxis+majorAxis)/(vertebraeLine*0.25);

	return vhs;
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

//assumes obj has a url pointing to s3 image, and an annotation object
async function writeRow(obj) {
	var tableBody = document.getElementById('eval-table').getElementsByTagName('tbody')[0]
	var row = tableBody.insertRow(tableBody.rows.length);


	let image = new Image();
	image.crossOrigin = 'anonymous';
	image.row = row;
	image.src = obj.url;
	image.style.width=IMAGE_SIZE + 'px';
	image.style.height=IMAGE_SIZE + 'px';

	image.onload = async function() {
		var gt_canvas = document.createElement('canvas');
		var pred_canvas = document.createElement('canvas');



		var img_cell = this.row.insertCell(0);
		img_cell.style.width = '250px'
		img_cell.appendChild(this);

		
		let gt_vhs = getVHS(obj.annotation);
		drawCanvas(gt_canvas, this, obj.annotation, pred=false);

		var gt_cell = this.row.insertCell(1);
		gt_cell.style.width='250px';
		gt_cell.appendChild(gt_canvas);


		let preds = await predict(this);

		for (let i=0;i<preds.length;i++) {
			preds[i] = preds[i] / COORD_SCALE;
		}

		let pred_vhs = getVHS(preds);
		drawCanvas(pred_canvas, this, preds);
	
		var pred_cell = this.row.insertCell(2);
		pred_cell.style.width = '250px';
		pred_cell.appendChild(pred_canvas);

		var acc_cell = this.row.insertCell(3);
		create_accuracy(obj.annotation, preds, acc_cell);

		pred_scores.push(pred_vhs);

	};

}

function distance(x1, y1, x2, y2) {
	let x = x2 - x1;
	let y = y2 - y1;
	return Math.sqrt(x*x + y*y);
}

function populate_individual_accuracy(gt, preds) {
	var tableBody = $('#accTableBody')[0];
	

	for (let i = 0; i<gt.length; i++) {
		var row = tableBody.insertRow(tableBody.rows.length);
		var label_cell = row.insertCell(0);
		label_cell.appendChild(document.createTextNode(POINT_LABELS[i]));

		var gt_cell = row.insertCell(1);
		gt_cell.appendChild(document.createTextNode(gt[i]));

		var pred_cell = row.insertCell(2);
		pred_cell.appendChild(document.createTextNode(preds[i]));

		var diff_cell = row.insertCell(3);
		diff_cell.appendChild(document.createTextNode(gt[i]-preds[i]));
	}
}

function create_accuracy(gt, preds, cell) {
	let gt_vertebrae = distance(gt[8], gt[9], gt[16], gt[17]);
	let pred_vertebrae = distance(preds[8], preds[9], preds[16], preds[17]);
	
	let gt_minor = distance(gt[4], gt[5], gt[6], gt[7]);
	let pred_minor = distance(preds[4], preds[5], preds[6], preds[7]);
	
	let gt_major = distance(gt[0], gt[1], gt[2], gt[3]);
	let pred_major = distance(preds[0], preds[1], preds[2], preds[3]);	

	let gt_vhs = (gt_minor+gt_major)/(gt_vertebrae*0.25);
	let pred_vhs = (pred_minor+pred_major)/(pred_vertebrae*0.25);
	

	var accContainer = document.createElement("div");
	var accTable = document.createElement("table");

	var headerRow = accTable.insertRow(0);
	var label_header = headerRow.insertCell(0);
	label_header.innerHTML = 'Label';

	var gt_header = headerRow.insertCell(1);
	gt_header.innerHTML = 'Ground Truth';

	var pred_header = headerRow.insertCell(2);
	pred_header.innerHTML = 'Model Prediction';


	var diff_header = headerRow.insertCell(3);
	diff_header.innerHTML = 'Difference :';

	var vertebraeRow = accTable.insertRow(1);
	vertebraeRow.insertCell(0).innerHTML = 'Vertebrae Length: ';
	vertebraeRow.insertCell(1).innerHTML = gt_vertebrae.toFixed(3);
	vertebraeRow.insertCell(2).innerHTML = pred_vertebrae.toFixed(3);
	vertebraeRow.insertCell(3).innerHTML = (pred_vertebrae - gt_vertebrae).toFixed(3);

	var minorAxisRow = accTable.insertRow(2);
	minorAxisRow.insertCell(0).innerHTML = 'Minor Axis Length: ';
	minorAxisRow.insertCell(1).innerHTML = gt_minor.toFixed(3);
	minorAxisRow.insertCell(2).innerHTML = pred_minor.toFixed(3);
	minorAxisRow.insertCell(3).innerHTML = (pred_minor - gt_vertebrae).toFixed(3);

	var majorAxisRow = accTable.insertRow(3);
	majorAxisRow.insertCell(0).innerHTML = 'Major Axis Length';
	majorAxisRow.insertCell(1).innerHTML = gt_major.toFixed(3);
	majorAxisRow.insertCell(2).innerHTML = pred_major.toFixed(3);
	majorAxisRow.insertCell(3).innerHTML = (pred_major - gt_major).toFixed(3);

	var vhsRow = accTable.insertRow(4);
	vhsRow.insertCell(0).innerHTML = 'VHS Score:';
	vhsRow.insertCell(1).innerHTML = gt_vhs.toFixed(3);
	vhsRow.insertCell(2).innerHTML = pred_vhs.toFixed(3);
	vhsRow.insertCell(3).innerHTML = (pred_vhs - gt_vhs).toFixed(3);

	var detailsButton = document.createElement("button");
	detailsButton.classList.add('btn');
	detailsButton.classList.add('btn-link');
	detailsButton.innerHTML = 'Show details';
	
	detailsButton.onclick = function() {

		$('#accTableBody').empty();
		populate_individual_accuracy(gt, preds);
		$("#points_dialog").dialog("open");

	}

	accContainer.appendChild(accTable);
	accContainer.appendChild(detailsButton);

	cell.appendChild(accContainer);
}

function fillTable(data) {
	var tableBody = document.getElementById('eval-table').getElementsByTagName('tbody')[0]

	for(let i = 0; i < 5; i++) {
		var row = tableBody.insertRow(tableBody.rows.length);
		var obj = data[i];
		gt_scores.push(getVHS(obj.annotation));

		writeRow(obj)
	}

}

setInterval(function() {

  if(gt_scores.length > 0 && gt_scores.length === pred_scores.length) {
  	let mean_error = 0;
  	for (let i =0; i< gt_scores.length; i++) {
  		mean_error = mean_error + Math.abs(gt_scores[i]-pred_scores[i]);
  	}

  	mean_error = mean_error / gt_scores.length;

  	$('#vhsmse').html(mean_error);
  }
}, 5000);


