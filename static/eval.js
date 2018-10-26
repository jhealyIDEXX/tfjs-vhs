const IMAGE_SIZE = 224;
const COORD_SCALE = 5.0;
let vhsModel = null;

$(document).ready(function() {
	loadModel();

	$.getJSON('eval_data.json', function(data) {
		fillTable(data);
	});
});


async function loadModel() {
	vhsModel = await tf.loadModel('models/vhs/model.json')
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

function drawVhs(ctx, points, x, y, pred=true) {
	let minorAxis = distance(points[4], points[5], points[6], points[7]);
	let majorAxis = distance(points[0], points[1], points[2], points[3]);
	let vertebraeLine = distance(points[8], points[9], points[16], points[17]);
	let vhs = (minorAxis+majorAxis)/(vertebraeLine*0.25);
	let text = ''
	if(pred) {
		text = 'Auto VHS: '+vhs.toFixed(2);
	} else {
		text = 'GT VHS: '+vhs.toFixed(2);
	}
	ctx.fillStyle = 'rgb(250, 250, 0)';
	ctx.fillText(text, x, y);

	return vhs.toFixed(2)
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

	let vhs = drawVhs(ctx, points, points[16]*w+20, points[17]*h, pred);

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

	return vhs
}

function writeRow(image, annotation) {
	var tableBody = document.getElementById('eval-table').getElementsByTagName('tbody')[0]
	var row = tableBody.insertRow(tableBody.rows.length);

	var img = document.createElement('img')
	img.src = image.src;
	img.style.height = IMAGE_SIZE+'px';
	img.style.width = IMAGE_SIZE+'px';
	var img_cell = row.insertCell(0);
	img_cell.style.width='250px';
	img_cell.appendChild(img);

	
	//draw ground truth
	var gt_canvas = document.createElement('canvas');
	gt_vhs = drawCanvas(gt_canvas, img, annotation, false);
	gt_cell = row.insertCell(1);
	gt_cell.style.width = '250px';
	gt_cell.appendChild(gt_canvas);

	var pred_canvas = document.createElement('canvas');
	preds = predict(img);
	pred_vhs = drawCanvas(pred_canvas, img, preds);

}

function fillTable(data) {
	var tableBody = document.getElementById('eval-table').getElementsByTagName('tbody')[0]

	for(let i = 0; i < data.length; i++) {
		var row = tableBody.insertRow(tableBody.rows.length);
		var obj = data[i];

		let image = new Image();
		image.crossOrigin = 'anonymous';
		image.height = IMAGE_SIZE
		image.width = IMAGE_SIZE
		image.annotation = obj.annotation
		image.src = obj.url;
		image.onload = function() {
			writeRow(this, this.annotation)
		}


	}
}

