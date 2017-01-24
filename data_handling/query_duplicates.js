//Modified from:
//http://stackoverflow.com/questions/14770170/how-to-find-mongo-documents-with-a-same-field Stennie's answer
db = db.getSiblingDB('dbtt')
cursor = db.ucsbivarplus.aggregate(
    { $group: { 
        // Group by fields to match on (a,b)
        _id: { Alloy: "$Alloy", flux_n_cm2_sec: "$flux_n_cm2_sec", fluence_n_cm2: "$fluence_n_cm2" },

        // Count number of matching docs for the group
        count: { $sum:  1 },

        // Save the _id for matching docs
        docs: { $push: "$_id" }
    }},

    // Limit results to duplicates (more than 1 match) 
    { $match: {
        count: { $gt : 1 }
    }}
)
while ( cursor.hasNext() ) {
    printjson( cursor.next() );
    }
