let express = require("express");
let app = express();

app.use(function(req, res, next){
	console.log(`${new Date()} - ${req.method} request for ${req.url}`);
	next();
});

app.use(express.static("../static"));

app.listen(8000, function(){
	console.log("serving at 8000")
})