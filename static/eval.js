const IMAGE_SIZE = 224;
const COORD_SCALE = 5.0;
var vhsModel = null;

$(document).ready(function() {
	loadModel();
});


async function loadModel() {
	console.log('model loaded');
	vhsModel =  await tf.loadModel('models/vhs/model.json');

	$.getJSON('eval_data.json', function(data) {
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
	console.log(values);
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
	ctx.fillText(text, 5, 15);

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

/*
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
*/
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

		
		gt_vhs = drawCanvas(gt_canvas, this, obj.annotation, pred=false);

		var gt_cell = this.row.insertCell(1);
		gt_cell.style.width='250px';
		gt_cell.appendChild(gt_canvas);


		let preds = await predict(this);

		for (let i=0;i<preds.length;i++) {
			preds[i] = preds[i] / COORD_SCALE;
		}

		pred_vhs = drawCanvas(pred_canvas, this, preds)

		/*
		let img = new Image();
		img.crossOrigin = 'anonymous';
		img.style.height = IMAGE_SIZE + 'px';
		img.style.width = IMAGE_SIZE + 'px';
		img.src = this.src;
		img.onload = function() {
			this.height = IMAGE_SIZE;
			this.width = IMAGE_SIZE;
			this.preds = predict(this);
			this.pred_vhs = drawCanvas(pred_canvas, this, this.preds)
		};
		*/
	
		var pred_cell = this.row.insertCell(2);
		pred_cell.style.width = '250px';
		pred_cell.appendChild(pred_canvas);

	};

}

function distance(x1, y1, x2, y2) {
	let x = x2 - x1;
	let y = y2 - y1;
	return Math.sqrt(x*x + y*y);
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
	

	get_point_accuracy(gt, preds)
}

function fillTable(data) {
	var tableBody = document.getElementById('eval-table').getElementsByTagName('tbody')[0]

	for(let i = 0; i < data.length; i++) {
		var row = tableBody.insertRow(tableBody.rows.length);
		var obj = data[i];

		writeRow(obj)
	}
	
}

